FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal; sentence-transformers/torch ship manylinux wheels.
RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the local embedding model into the image so retrieval works fully offline
# after the build (no model download at container start).
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# After the model is cached, force offline mode so the runtime never makes a Hub
# network call (no startup latency / failures in locked-down environments).
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY app ./app
COPY static ./static
COPY data ./data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
