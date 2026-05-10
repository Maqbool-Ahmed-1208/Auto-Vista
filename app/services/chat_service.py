from fastapi.concurrency import run_in_threadpool

# from app.core.clients import embedding_client, index, groq_client
from openai import OpenAI
from app.core.clients import index
from groq import Groq
from app.core.config import LLM_MODEL, PINECONE_NAMESPACE, NVIDIA_API, GROQ_API_KEY
import logging


# ------------------------
# NVIDIA Embedding API
# ------------------------
def _get_embedding(text="None"):
    """Blocking call to get NVIDIA embedding for a text string."""
    # NVIDIA embedding client
    embedding_client = OpenAI(
        api_key=NVIDIA_API,
        base_url="https://integrate.api.nvidia.com/v1",
    )

    response = embedding_client.embeddings.create(
        input=text,
        model="nvidia/nv-embed-v1",
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "NONE"},
    )
    return response.data[0].embedding


async def get_embedding_async(text="None"):
    """Async wrapper for embedding call."""
    return await run_in_threadpool(_get_embedding, text)


# ------------------------
# Pinecone Query
# ------------------------
def _query_pinecone(embedding):
    """Blocking Pinecone query."""
    result = index.query_namespaces(
        vector=embedding,
        namespaces=PINECONE_NAMESPACE,
        metric="cosine",
        top_k=25,
        include_metadata=True,
    )
    return result["matches"]


async def query_pinecone_async(embedding):
    """Async wrapper for Pinecone query."""
    return await run_in_threadpool(_query_pinecone, embedding)


# Instantiate client once and reuse across calls
client = Groq(api_key=GROQ_API_KEY)


def groq_chunk_cleaner(chunk: str) -> str:
    """
    Cleans a text chunk using Groq's model by stripping formatting,
    sources, and boilerplate sections.
    Preserves the exact user wording and phrasing of the main content.
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Text Cleaning Assistant.\n"
                        "Your ONLY job is to return the exact same text as provided by the user, "
                        "but cleaned of unwanted elements.\n\n"
                        "Strict rules:\n"
                        "- DO NOT rephrase, paraphrase, or summarize the wording.\n"
                        "- Preserve all original sentences, casing, punctuation, and wording of the main text.\n"
                        "- Remove all formatting (Markdown, HTML, LaTeX, bullet points, headers, etc.).\n"
                        "- Remove links, URLs, and citations.\n"
                        "- Remove boilerplate sections such as:\n"
                        "   * 'Sources:' and everything after it.\n"
                        "   * 'Follow-up Question Suggestions:' and everything after it.\n"
                        "- Output only the cleaned plain text content with no extra commentary."
                    ),
                },
                {
                    "role": "user",
                    "content": chunk,
                },
            ],
            temperature=0.1,  # strictly deterministic
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        logging.error("Groq text cleaning failed: %s", e)
        return "[Error] Unable to process the request at the moment."


def pinecone_query_maker(user_query, history):
    """
    Generates an optimized prompt from Groq using prior history and current input.
    If the input is unrelated, history is ignored. Follow-up logic is handled externally.
    """
    client = Groq(api_key=GROQ_API_KEY)
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a prompt optimization engine. Your task is to generate a clean, context-aware prompt "
                        "for querying a vector database or a language model.\n\n"
                        "You are given:\n"
                        "- Prior conversation history (which may or may not be relevant)\n"
                        "- A new user input (current query)\n\n"
                        "Instructions:\n"
                        "- If the user input is clearly related to the conversation history, merge the relevant context "
                        "to enrich and clarify the prompt.\n"
                        "- If the new input is unrelated or self-contained, do **not** incorporate history—just enhance the "
                        "standalone query for precision and clarity.\n"
                        "- If user input contains 'no', 'nah', 'not', 'nope', or 'nopes', respond only with: 'no'.\n"
                        "- If user input contains 'yes', 'yup', 'yo', or 'y', analyze the prior question or topic in the Conversation History and generate a meaningful follow-up prompt based on it.\n\n"
                        "Return only the final optimized user prompt as plain text—no extra commentary, headers, or formatting."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Conversation History:\n{history.strip()}\n\n"
                        f"User Input:\n{user_query.strip()}"
                    ),
                },
            ],
            model=LLM_MODEL,
            temperature=0.4,
            stream=False,
        )

        return chat_completion.choices[0].message.content.strip()

    except Exception as e:
        print("Groq streaming failed:", e)
        return "[Error] Unable to process the request at the moment."


def _query_groq_stream(user_input, relevant_context):
    """Blocking Groq streaming call."""
    groq_client = Groq(api_key=GROQ_API_KEY)
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"{relevant_context}"},
                {"role": "user", "content": f"{user_input}"},
            ],
            model=LLM_MODEL,
            temperature=0.3,
            stream=True,
        )

        for chunk in chat_completion:
            content = chunk.choices[0].delta.content or ""
            yield content

    except Exception as e:
        print("Groq streaming failed:", e)
        yield "[Error] Unable to process the request at the moment."


async def query_groq_stream_async(user_input, relevant_context):
    """Async wrapper for Groq streaming."""
    generator = _query_groq_stream(user_input, relevant_context)
    for chunk in generator:
        yield chunk
