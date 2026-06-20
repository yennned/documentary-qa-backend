#!/bin/bash
# Start the Ollama server, then pull the chat (and optionally embedding) models on
# first boot so `docker compose up` is the only command a reviewer needs. Pulled models
# live on a named volume, so subsequent starts are instant and fully offline.
#
# Deliberately NOT using `set -e`: a transient pull failure must not kill PID 1 and
# leave the server orphaned with the healthcheck never satisfied. We retry instead.

ollama serve &
server_pid=$!

# Forward termination signals to the server for a graceful `docker compose down`.
trap 'kill -TERM "$server_pid" 2>/dev/null; wait "$server_pid"; exit 0' TERM INT

echo "Waiting for Ollama to be ready..."
until ollama list >/dev/null 2>&1; do
  sleep 1
done

pull_with_retry() {
  local model="$1"
  local attempt=1
  until ollama pull "$model"; do
    if [ "$attempt" -ge 5 ]; then
      echo "WARNING: failed to pull ${model} after ${attempt} attempts; will keep serving and retry on next boot." >&2
      return 0
    fi
    echo "Pull of ${model} failed (attempt ${attempt}); retrying in 5s..." >&2
    attempt=$((attempt + 1))
    sleep 5
  done
}

CHAT_MODEL="${OLLAMA_MODEL:-llama3.1:8b}"
echo "Pulling chat model: ${CHAT_MODEL}"
pull_with_retry "${CHAT_MODEL}"

# Only needed if you switch embeddings to the Ollama backend (EMBED_BACKEND=ollama).
if [ "${PULL_EMBED_MODEL:-false}" = "true" ]; then
  EMBED_MODEL_NAME="${EMBED_MODEL:-nomic-embed-text}"
  echo "Pulling embedding model: ${EMBED_MODEL_NAME}"
  pull_with_retry "${EMBED_MODEL_NAME}"
fi

echo "Models ready."
wait "${server_pid}"
