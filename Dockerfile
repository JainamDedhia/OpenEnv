# ── Base image ────────────────────────────────────────────
FROM python:3.11-slim

# HF Spaces requires port 7860
EXPOSE 7860

# ── System deps ───────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── App source ────────────────────────────────────────────
COPY . .

# ── Runtime environment variables (override via HF Secrets)
ENV API_BASE_URL="https://openrouter.ai/api/v1"
ENV MODEL_NAME="openai/gpt-4o-mini"
ENV HF_TOKEN=""

# ── Start FastAPI server ───────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]