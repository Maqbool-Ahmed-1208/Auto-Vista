from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api import chat
# from app.api import chat, transcribe, speak

app = FastAPI(title="AutoVista")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific frontend domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Routers
app.include_router(chat.router, prefix="/chat", tags=["Chat"])

# app.include_router(transcribe.router, prefix="/stt")

# app.include_router(speak.router, prefix="/tts")

print("✅ FastAPI backend for AutoVista AI Assistant is ready.")

# print("🌐 Visit http://127.0.0.1:8000 to open chat UI.")
# print("📄 API docs available at http://127.0.0.1:8000/docs")

print("🌐 Visit https://inveros-tech-RAG-AUTO-VISTA.hf.space/chat to open chat UI.")
print("📄 API docs available at https://inveros-tech-RAG-AUTO-VISTA.hf.space/docs")
