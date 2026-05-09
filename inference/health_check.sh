#!/usr/bin/env bash
# ============================================================================
# FinAgent Inference Health Check
# Validates the vLLM inference endpoint is operational and responding correctly.
#
# Usage:
#   ./health_check.sh [--host HOST] [--port PORT] [--timeout SECONDS]
#
# Exit Codes:
#   0 - Success: endpoint healthy, latency reported
#   1 - Connection error (refused or network unreachable)
#   2 - Timeout (no response within timeout period)
#   3 - Non-200 HTTP response
#   4 - Empty response body
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# --- CLI Argument Parsing ---
HOST="${DEFAULT_HOST}"
PORT="${DEFAULT_PORT}"
TIMEOUT="${HEALTH_CHECK_TIMEOUT}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

BASE_URL="http://${HOST}:${PORT}"

# --- Helper: Handle curl exit codes ---
# Checks curl's exit code and prints appropriate error messages.
# Arguments: $1 = curl exit code, $2 = context description
handle_curl_error() {
    local curl_exit="$1"
    local context="$2"

    case "${curl_exit}" in
        0)
            # Success, no error
            return 0
            ;;
        7)
            echo "Health check failed: Connection refused at ${HOST}:${PORT}" >&2
            exit 1
            ;;
        6)
            echo "Health check failed: Network unreachable" >&2
            exit 1
            ;;
        28)
            echo "Health check failed: Timeout: no response within ${TIMEOUT}s" >&2
            exit 2
            ;;
        *)
            echo "Health check failed: ${context} (curl exit code: ${curl_exit})" >&2
            exit 1
            ;;
    esac
}

# --- Detect Loaded Model Name ---
set +e
models_response=$(curl -s --max-time "${TIMEOUT}" "${BASE_URL}/v1/models" 2>/dev/null)
curl_exit=$?
set -e

handle_curl_error "${curl_exit}" "Failed to reach /v1/models endpoint"

MODEL_NAME=$(echo "${models_response}" | jq -r '.data[0].id')

# --- Send Health Check Request ---
payload=$(jq -n \
    --arg model "${MODEL_NAME}" \
    '{
        model: $model,
        messages: [{role: "user", content: "Say hello."}],
        max_tokens: 50,
        temperature: 0.1
    }')

# Measure latency using curl's time_total write-out
set +e
response=$(curl -s -w "\n%{http_code}\n%{time_total}" \
    --max-time "${TIMEOUT}" \
    -X POST \
    -H "Content-Type: application/json" \
    -d "${payload}" \
    "${BASE_URL}/v1/chat/completions" 2>/dev/null)
curl_exit=$?
set -e

handle_curl_error "${curl_exit}" "Failed to reach /v1/chat/completions endpoint"

# --- Parse Response ---
# Extract HTTP status code and timing from curl output
http_code=$(echo "${response}" | tail -n 1 | tr -d '[:space:]')
time_total=$(echo "${response}" | tail -n 2 | head -n 1 | tr -d '[:space:]')
body=$(echo "${response}" | sed '$d' | sed '$d')

# --- Validate Response ---
# Check HTTP 200
if [[ "${http_code}" != "200" ]]; then
    echo "Health check failed: HTTP ${http_code}" >&2
    echo "${body}" >&2
    exit 3
fi

# Extract content from choices[0].message.content
content=$(echo "${body}" | jq -r '.choices[0].message.content')

# Validate non-empty content
if [[ -z "${content}" || "${content}" == "null" ]]; then
    echo "Health check failed: Empty response from model" >&2
    exit 4
fi

# --- Report Success ---
# Convert time_total (seconds as float) to milliseconds
latency_ms=$(echo "${time_total}" | awk '{printf "%.0f", $1 * 1000}')

echo "Health check passed: model=${MODEL_NAME}, latency=${latency_ms}ms"
exit 0
