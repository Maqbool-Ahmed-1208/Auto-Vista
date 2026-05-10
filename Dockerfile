FROM python:3.11-slim-bookworm

LABEL maintainer="Maqbool Ahmed <maqboolshaikh321@gmail.com>"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONWARNINGS="ignore:Unverified HTTPS request"


# Install dependencies for PyAudio
RUN apt-get update && apt-get install -y \
    gcc \
    libasound2-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip==24.3.1

# Create workdir
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies (uvicorn[standard] includes websockets)
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "uvicorn[standard]"

# Copy app code and assets
COPY app ./app
# COPY *.png ./

# Expose Hugging Face Spaces port (must be 7860)
EXPOSE 7860

# Start FastAPI with WebSocket support
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--ws", "websockets"]
