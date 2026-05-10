import os
from dotenv import load_dotenv

# Load variables from .env file (if present)
load_dotenv()

# Application mode
APPLICATION_TYPE = os.getenv("APPLICATION_TYPE", "STANDALONE")
print(f"APPLICATION_TYPE = {APPLICATION_TYPE}")

# === PINECONE ===
PINECONE_API = os.getenv("PINECONE_API")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "").split(",") if os.getenv("PINECONE_NAMESPACE") else [
    "cars",
]

# === GROQ ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")   # instead of TEST_API
LLM_MODEL = os.getenv("LLM_MODEL")

# === NVIDIA ===
NVIDIA_API = os.getenv("NVIDEA_EMBEDDING_API")
# NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL")

# === ASSEMBLY ===
ASSEMBLY_API_KEY = os.getenv("ASSEMBLY_API_KEY")

# === CARTESIA ===
# CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
