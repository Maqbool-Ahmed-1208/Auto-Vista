from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from app.models.chat import ChatRequest
from app.services.chat_service import (
    get_embedding_async,
    query_pinecone_async,
    query_groq_stream_async,
    pinecone_query_maker,
)
import json
import uuid

router = APIRouter()

# In-memory session store (for dev only)
chat_sessions = {}


@router.get("/")
def serve_chat_html():
    """Serve chat.html from static directory."""
    return FileResponse("app/static/chat.html")


@router.post("/stream/start")
async def start_chat_session(payload: ChatRequest):
    """Initialize a new chat session."""
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = {
        "user_query": payload.user_query,
        "chat_state": payload.chat_state,
        "memory_state": payload.memory_state,
    }
    return {"session_id": session_id}


@router.get("/stream")
async def chat_stream(session_id: str):
    """Stream AI response for a session using SSE."""
    session = chat_sessions.pop(session_id, None)
    if not session:
        return JSONResponse({"error": "Invalid session_id"}, status_code=400)

    user_query = session["user_query"]
    chat_state = session["chat_state"]
    memory_state = session["memory_state"]

    async def event_stream():
        partial_response = ""
        # Build chat history from memory_state

        history_str = "\n".join(memory_state)
        if len(memory_state) > 0:
            history = f"{memory_state[-1]}"
        else:
            history = f"{memory_state}"

        pinecone_query = pinecone_query_maker(user_query, history)

        # Async embedding
        embedding = await get_embedding_async(pinecone_query)

        # Async Pinecone query
        relevant_chunks = await query_pinecone_async(embedding)

        # Sort all chunks in descending order by score
        sorted_chunks = sorted(relevant_chunks, key=lambda x: x["score"], reverse=True)

        # Select the top 6 scoring chunks
        top_chunks = sorted_chunks[:6]

        # Build context from top chunks
        context = "\n".join(
            f"{chunk['score']}\n"
            f"{chunk['metadata'].get('source', '')}\n"
            f"{chunk['metadata'].get('link', '')}\n"
            f"{chunk['metadata'].get('text', '')}"
            for chunk in top_chunks
        )

        prompt = f"""
        You are **AutoVista**, a professional automotive assistant.

        🎯 Use *only* the information from the provided document context to respond.

        🚫 DO NOT:
        - Invent, assume, or infer beyond what's explicitly stated in the context.
        - Generate or modify any links—only use links already present in the **context**.
        - Reference external sources or extrapolate outside the provided material.

        📌 Note:
        If the user replies with "no", "nah", "not", "nope", or "nopes", respond only with:
        **"Anything else I can help you with?"**

        ---

        📄 **Context**:
        {context}

        ---

        🧠 **Response Guidelines**:
        - ✅ Format all outputs in **Markdown** for readability.
        - ✅ Convert any tables into well-structured paragraphs for smoother narrative flow.
        - ✅ Conclude every response with the following clearly labeled sections:

        ### Follow-Up Question Suggestions:
        *Would you like to ask any of these follow-up questions based on the above?*

        ### Sources:
        - [Document Name](Link)

        - ✅ If the context contains no relevant information, respond with:
        *"No relevant information available in the provided documents."*

        - ✅ Never fabricate sources, assumptions, or external references.
        """

        # Async LLM streaming with full context
        async for chunk in query_groq_stream_async(pinecone_query, prompt):
            # async for chunk in query_groq_stream_async(user_prompt, prompt):
            partial_response += chunk
            data = {
                "partial_response": partial_response,
                "chat_state": chat_state + [(user_query, partial_response)],
                "memory_state": memory_state,
            }
            yield f"data: {json.dumps(data)}\n\n"

        # Final state update
        memory_state.append(f"User: {user_query}")
        memory_state.append(f"{partial_response}")
        chat_state.append((user_query, partial_response))

        yield f"data: {json.dumps({'final': True, 'chat_state': chat_state, 'memory_state': memory_state})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/clear")
async def clear_chat():
    """Clear session state (frontend reset)."""
    return JSONResponse(content={"chat_state": [], "memory_state": [], "input_box": ""})
