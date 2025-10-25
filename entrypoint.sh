#!/usr/bin/env bash
# entrypoint: convert environment variables into CLI args for whisperworker.py
set -euo pipefail

# Required environment variables
: "${SERVER_URL:?Need to set SERVER_URL environment variable (e.g. https://example.com)}"
: "${USERNAME:?Need to set USERNAME environment variable}"
: "${PASSWORD:?Need to set PASSWORD environment variable}"
: "${WORKER_NAME:?Need to set WORKER_NAME environment variable}"

# Optional
MAX_CONCURRENT_VALUE="${MAX_CONCURRENT:-}"

# Build args array
args=(--server-url "$SERVER_URL" --username "$USERNAME" --password "$PASSWORD" --worker-name "$WORKER_NAME")
if [ -n "$MAX_CONCURRENT_VALUE" ]; then
  args+=(--max-concurrent "$MAX_CONCURRENT_VALUE")
fi

# If any extra args were provided to the container, append them
if [ "$#" -gt 0 ]; then
  args+=("$@")
fi

echo "Starting whisperworker with: ${args[*]}"
exec python -u whisperworker.py "${args[@]}"
