from pinecone import Pinecone
from openai import OpenAI
from groq import Groq
from cartesia import Cartesia
from app.core.config import PINECONE_API, NVIDIA_API, GROQ_API_KEY
# from app.core.config import PINECONE_API, NVIDIA_API

# Pinecone client
pc = Pinecone(api_key=PINECONE_API)

index = pc.Index("suzuki")

# NVIDIA embedding client
embedding_client = OpenAI(
    api_key=NVIDIA_API,
    base_url="https://integrate.api.nvidia.com/v1",
)

# Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Cartesia client
# cartesia_client = Cartesia(api_key=CARTESIA_API_KEY)

