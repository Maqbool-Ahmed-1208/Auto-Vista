# app/services/assembly_bridge.py
import json
import threading
import queue
import time
from datetime import datetime
from urllib.parse import urlencode
from websocket import WebSocketApp, ABNF
import asyncio
from fastapi import WebSocket
from app.core.config import ASSEMBLY_API_KEY
from app.models.chat import STTResponse

SAMPLE_RATE = 16000
API_URL = f"wss://streaming.assemblyai.com/v3/ws?{urlencode({'sample_rate': SAMPLE_RATE, 'format_turns': True})}"
HEARTBEAT_INTERVAL = 2 * 5
TERMINATION_WAIT = 2 * 0.75  # seconds to wait for assembly final messages


class AssemblyBridge:
    """
    Threaded bridge that proxies audio to AssemblyAI and forwards Assembly messages
    back to the client WebSocket (self.client_ws). Designed to be created/destroyed
    per speech burst while keeping the client WS connection alive.
    """

    def __init__(self, client_ws: WebSocket, loop: asyncio.AbstractEventLoop):
        self.client_ws = client_ws
        self.loop = loop
        self.audio_q: queue.Queue = queue.Queue(maxsize=80)
        self.stop_evt = threading.Event()
        self.terminated_evt = threading.Event()
        self.assembly_ws_app: WebSocketApp | None = None
        self.sender_thread: threading.Thread | None = None
        self.heartbeat_thread: threading.Thread | None = None

        # transcript state (per bridge)
        self.final_transcript: str = ""
        self.last_transcript: str = ""
        self.chunks: list[str] = []
        self.last_chunk_time: datetime | None = None

        # internal flags
        self._termination_requested = False

    # ------- start / threads -------

    def start(self) -> None:
        """Starts the AssemblyAI websocket in a new thread."""
        headers = [f"Authorization: {ASSEMBLY_API_KEY}"]
        self.assembly_ws_app = WebSocketApp(
            API_URL,
            header=headers,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self.sender_thread = threading.Thread(
            target=self.assembly_ws_app.run_forever, daemon=True
        )
        self.sender_thread.start()

    def _sender(self, ws: WebSocketApp) -> None:
        """Sends binary audio frames from queue to Assembly."""
        try:
            while not self.stop_evt.is_set():
                try:
                    chunk = self.audio_q.get(timeout=0.3)
                except queue.Empty:
                    continue
                if chunk is None:  # sentinel to break
                    break
                try:
                    ws.send(chunk, opcode=ABNF.OPCODE_BINARY)
                    self.last_chunk_time = datetime.now()
                except Exception:
                    self.stop_evt.set()
                    break
        except Exception:
            self.stop_evt.set()

    def _heartbeat(self, ws: WebSocketApp) -> None:
        try:
            while not self.stop_evt.is_set():
                time.sleep(HEARTBEAT_INTERVAL)
                try:
                    if ws and ws.sock and ws.sock.connected:
                        ws.send(json.dumps({"type": "Ping"}))
                except Exception:
                    break
        except Exception:
            pass

    # ------- websocket callbacks (assembly) -------

    def _on_open(self, ws: WebSocketApp) -> None:
        # start sender + heartbeat threads
        threading.Thread(target=self._sender, args=(ws,), daemon=True).start()
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat, args=(ws,), daemon=True
        )
        self.heartbeat_thread.start()
        # reset session-level variables
        self._termination_requested = False
        self.final_transcript = ""
        self.last_transcript = ""
        self.chunks.clear()
        self.last_chunk_time = datetime.now()
        self.terminated_evt.clear()

    def _on_message(self, ws: WebSocketApp, message: str) -> None:
        # forward raw assembly json upstream (optional)
        try:
            # forward AssemblyAI raw message string to client
            asyncio.run_coroutine_threadsafe(
                self.client_ws.send_text(message), self.loop
            )
        except Exception:
            # If we can't forward, mark stop so the bridge will close
            self.stop_evt.set()
            self.terminated_evt.set()
            return

        try:
            data = json.loads(message)
        except Exception:
            return

        mtype = data.get("type")

        # Handle session begin
        if mtype == "Begin":
            self.last_chunk_time = datetime.now()
            self.chunks.clear()
            self.final_transcript = ""
            self.last_transcript = ""

        elif mtype == "Turn":
            transcript = (data.get("transcript") or "").strip()

            # Only take the new part (delta) since last update
            if transcript.startswith(self.last_transcript):
                new_text = transcript[len(self.last_transcript) :]
            else:
                new_text = transcript

            if new_text.strip():
                self.chunks.append(new_text.strip())
                self.final_transcript = new_text.strip() + " "
                asyncio.run_coroutine_threadsafe(
                    self.client_ws.send_text(
                        STTResponse(
                            type="partial",
                            text=new_text.strip(),
                            timestamp=datetime.utcnow().isoformat(),
                        ).model_dump_json()
                    ),
                    self.loop,
                )

            self.last_transcript = transcript
            self.last_chunk_time = datetime.utcnow()

        elif mtype == "Termination":
            asyncio.run_coroutine_threadsafe(
                self.client_ws.send_text(
                    STTResponse(
                        type="final",
                        text=self.final_transcript.strip(),
                        timestamp=datetime.utcnow().isoformat(),
                    ).model_dump_json()
                ),
                self.loop,
            )

    def _on_error(self, ws: WebSocketApp, error: Exception) -> None:
        try:
            asyncio.run_coroutine_threadsafe(
                self.client_ws.send_text(
                    STTResponse(type="error", text=str(error)).model_dump_json()
                ),
                self.loop,
            )
        except Exception:
            pass
        self.stop_evt.set()
        self.terminated_evt.set()

    def _on_close(self, ws: WebSocketApp, code: int, reason: str) -> None:
        try:
            asyncio.run_coroutine_threadsafe(
                self.client_ws.send_text(
                    STTResponse(
                        type="closed", code=code, reason=reason
                    ).model_dump_json()
                ),
                self.loop,
            )
        except Exception:
            pass
        self.stop_evt.set()
        self.terminated_evt.set()

    # ------- helpers used by async route -------

    def queue_audio(self, chunk: bytes) -> None:
        """Non-blocking enqueue of audio chunk."""
        if self.stop_evt.is_set():
            return
        try:
            self.audio_q.put(chunk, block=False)
        except queue.Full:
            # drop silently (or log)
            pass

    def request_terminate(self) -> None:
        """
        Ask AssemblyAI to flush (send Terminate). This is best-effort and non-blocking.
        Caller should await a short time to give Assembly a chance to respond.
        """
        if self._termination_requested:
            return
        self._termination_requested = True
        try:
            if (
                self.assembly_ws_app
                and self.assembly_ws_app.sock
                and self.assembly_ws_app.sock.connected
            ):
                self.assembly_ws_app.send(json.dumps({"type": "Terminate"}))
        except Exception:
            pass

    def terminate(self) -> None:
        """
        Full cleanup (blocking). Safe to call from a background executor so it doesn't block event loop.
        Waits briefly for termination signal from Assembly (TERMINATION_WAIT) before closing sockets.
        """
        # If we already got a Termination message, terminated_evt will be set.
        # If not, we attempt to request Terminate (if not already requested),
        # then wait up to TERMINATION_WAIT for Assembly to respond.
        try:
            if not self._termination_requested:
                try:
                    if (
                        self.assembly_ws_app
                        and self.assembly_ws_app.sock
                        and self.assembly_ws_app.sock.connected
                    ):
                        self.assembly_ws_app.send(json.dumps({"type": "Terminate"}))
                except Exception:
                    pass
            waited = 0.0
            step = 0.05
            while waited < TERMINATION_WAIT:
                if self.terminated_evt.is_set():
                    break
                time.sleep(step)
                waited += step
        finally:
            self.stop_evt.set()
            try:
                self.audio_q.put_nowait(None)
            except Exception:
                pass
            try:
                if (
                    self.assembly_ws_app
                    and self.assembly_ws_app.sock
                    and self.assembly_ws_app.sock.connected
                ):
                    self.assembly_ws_app.close()
            except Exception:
                pass
            if self.sender_thread:
                self.sender_thread.join(timeout=1.0)
            if self.heartbeat_thread:
                self.heartbeat_thread.join(timeout=0.5)
