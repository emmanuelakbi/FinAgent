#!/usr/bin/env bash
# ============================================================================
# FinAgent Inference Setup Script
# Automates the full installation sequence from a fresh MI300X instance
# to a running vLLM inference server with health check verification.
#
# Usage:
#   ./setup.sh [--model MODEL] [--host HOST] [--port PORT] [--max-model-len LEN]
#
# Exit Codes:
#   0 - Success: server running and health check passed
#   1 - General failure (install error, invalid model)
#   2 - GPU not detected
#   3 - Model load failure (OOM)
#   4 - Health check failure
#   5 - Timeout
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# =============================================================================
# CLI Argument Parsing
# =============================================================================
# Defaults come from config.sh (which already respects environment variables
# FINAGENT_MODEL, FINAGENT_HOST, FINAGENT_PORT, FINAGENT_MAX_LEN)
MODEL="${DEFAULT_MODEL}"
HOST="${DEFAULT_HOST}"
PORT="${DEFAULT_PORT}"
MAX_MODEL_LEN="${DEFAULT_MAX_MODEL_LEN}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --model requires a value" >&2
                exit 1
            fi
            MODEL="$2"
            shift 2
            ;;
        --host)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --host requires a value" >&2
                exit 1
            fi
            HOST="$2"
            shift 2
            ;;
        --port)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --port requires a value" >&2
                exit 1
            fi
            PORT="$2"
            shift 2
            ;;
        --max-model-len)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --max-model-len requires a value" >&2
                exit 1
            fi
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./setup.sh [--model MODEL] [--host HOST] [--port PORT] [--max-model-len LEN]"
            echo ""
            echo "Options:"
            echo "  --model          HuggingFace model ID (default: Qwen/Qwen3-8B)"
            echo "  --host           Bind address (default: 0.0.0.0)"
            echo "  --port           Listening port (default: 8000)"
            echo "  --max-model-len  Maximum context length in tokens (default: 32768)"
            echo ""
            echo "Environment variables:"
            echo "  FINAGENT_MODEL   Override default model"
            echo "  FINAGENT_HOST    Override default host"
            echo "  FINAGENT_PORT    Override default port"
            echo "  FINAGENT_MAX_LEN Override default max model length"
            echo ""
            echo "Supported models:"
            for name in "${SUPPORTED_MODEL_NAMES[@]}"; do
                echo "  - ${name}"
            done
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'" >&2
            echo "Run './setup.sh --help' for usage information." >&2
            exit 1
            ;;
    esac
done

# =============================================================================
# Model Validation
# =============================================================================
validate_model() {
    local model_name="$1"
    for supported in "${SUPPORTED_MODEL_NAMES[@]}"; do
        if [[ "${model_name}" == "${supported}" ]]; then
            return 0
        fi
    done
    return 1
}

if ! validate_model "${MODEL}"; then
    echo "Error: Invalid model '${MODEL}'" >&2
    echo "Supported models:" >&2
    for name in "${SUPPORTED_MODEL_NAMES[@]}"; do
        echo "  - ${name}" >&2
    done
    exit 1
fi

# =============================================================================
# Configuration Summary
# =============================================================================
echo "============================================"
echo " FinAgent Inference Setup"
echo "============================================"
echo " Model:         ${MODEL}"
echo " Host:          ${HOST}"
echo " Port:          ${PORT}"
echo " Max Model Len: ${MAX_MODEL_LEN}"
echo "============================================"
echo ""

# =============================================================================
# GPU Detection
# =============================================================================
detect_gpu() {
    echo "Step: GPU Detection"

    if ! command -v rocm-smi &>/dev/null; then
        echo "Error: rocm-smi not found — GPU cannot be detected" >&2
        exit "${EXIT_GPU_NOT_DETECTED}"
    fi

    if ! rocm-smi --showid &>/dev/null; then
        echo "Error: rocm-smi found no GPU devices" >&2
        exit "${EXIT_GPU_NOT_DETECTED}"
    fi

    echo "  GPU detected successfully"
}

# =============================================================================
# ROCm Installation
# =============================================================================
install_rocm() {
    echo "Step: ROCm Installation"

    # Idempotency: check if ROCm is already installed at the correct version
    if command -v rocm-smi &>/dev/null; then
        local installed_version
        installed_version=$(rocm-smi --version 2>/dev/null | grep -oP '\d+\.\d+' | head -1 || true)
        if [[ "${installed_version}" == "${ROCM_VERSION}" ]]; then
            echo "  ROCm ${ROCM_VERSION} already installed, skipping"
            return 0
        fi
    fi

    echo "  Installing ROCm ${ROCM_VERSION}..."

    # Install ROCm kernel driver, runtime, and rocm-smi
    if ! sudo apt-get update -y 2>&1; then
        echo "Error: Failed to update package lists during ROCm installation" >&2
        exit "${EXIT_GENERAL_FAILURE}"
    fi

    if ! sudo apt-get install -y \
        "amdgpu-dkms" \
        "rocm-dev" \
        "rocm-smi-lib" 2>&1; then
        echo "Error: Failed to install ROCm ${ROCM_VERSION} packages" >&2
        exit "${EXIT_GENERAL_FAILURE}"
    fi

    # Verify installation succeeded
    if ! command -v rocm-smi &>/dev/null; then
        echo "Error: ROCm installation completed but rocm-smi not found" >&2
        exit "${EXIT_GENERAL_FAILURE}"
    fi

    echo "  ROCm ${ROCM_VERSION} installed successfully"
}

# =============================================================================
# Health Check Integration
# =============================================================================
run_health_check() {
    echo "Step: Health Check"

    local health_exit=0
    "${SCRIPT_DIR}/health_check.sh" --host "${HOST}" --port "${PORT}" || health_exit=$?

    if [[ ${health_exit} -eq 0 ]]; then
        echo "  Health check passed"
        return 0
    fi

    # Map any non-zero health check exit code to setup exit code 4
    echo "Error: Health check failed (health_check.sh exit code: ${health_exit})" >&2
    return 1
}

# =============================================================================
# Final Summary
# =============================================================================
print_summary() {
    local status="$1"

    echo ""
    echo "============================================"
    if [[ "${status}" == "success" ]]; then
        echo " FinAgent Inference Setup: SUCCESS"
        echo "============================================"
        echo " Model:    ${MODEL}"
        echo " Endpoint: http://${HOST}:${PORT}/v1/chat/completions"
        echo " Status:   Running and healthy"
        echo "============================================"
    else
        echo " FinAgent Inference Setup: FAILED"
        echo "============================================"
        echo " Model:    ${MODEL}"
        echo " Endpoint: http://${HOST}:${PORT}/v1/chat/completions"
        echo " Status:   Health check failed"
        echo "============================================"
    fi
}

# =============================================================================
# Execute Setup Steps
# =============================================================================
detect_gpu
install_rocm

# =============================================================================
# PyTorch Installation (ROCm 6.2-compatible)
# =============================================================================
install_pytorch() {
    echo "Step: PyTorch Installation"

    # Idempotency check: skip if correct version already installed
    local installed_version
    installed_version=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")

    if [[ "${installed_version}" == "${TORCH_VERSION}" ]]; then
        echo "  PyTorch ${TORCH_VERSION} already installed, skipping"
        return 0
    fi

    echo "  Installing PyTorch ${TORCH_VERSION}..."
    if ! pip install --no-cache-dir "torch==${TORCH_VERSION}" \
        --index-url "https://download.pytorch.org/whl/rocm6.2" 2>&1; then
        echo "Error: PyTorch installation failed" >&2
        echo "Step: PyTorch Installation" >&2
        exit "${EXIT_GENERAL_FAILURE}"
    fi

    # Verify GPU access
    echo "  Verifying PyTorch GPU access..."
    if ! python -c "import torch; assert torch.cuda.is_available(), 'GPU not available via PyTorch'" 2>&1; then
        echo "Error: PyTorch installed but GPU not accessible via torch.cuda.is_available()" >&2
        echo "Step: PyTorch Installation" >&2
        exit "${EXIT_GENERAL_FAILURE}"
    fi

    echo "  PyTorch ${TORCH_VERSION} installed and GPU access verified."
}

# =============================================================================
# vLLM Installation (ROCm backend)
# =============================================================================
install_vllm() {
    echo "Step: vLLM Installation"

    # Idempotency check: skip if correct version already installed
    local installed_version
    installed_version=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "")

    if [[ "${installed_version}" == "${VLLM_VERSION}" ]]; then
        echo "  vLLM ${VLLM_VERSION} already installed, skipping"
        return 0
    fi

    echo "  Installing vLLM ${VLLM_VERSION}..."
    if ! pip install --no-cache-dir "vllm==${VLLM_VERSION}" 2>&1; then
        echo "Error: vLLM installation failed" >&2
        echo "Step: vLLM Installation" >&2
        exit "${EXIT_GENERAL_FAILURE}"
    fi

    # Verify installation
    if ! python -c "import vllm; assert vllm.__version__ == '${VLLM_VERSION}'" 2>&1; then
        echo "Error: vLLM installed but version mismatch" >&2
        echo "Step: vLLM Installation" >&2
        exit "${EXIT_GENERAL_FAILURE}"
    fi

    echo "  vLLM ${VLLM_VERSION} installed successfully."
}

# =============================================================================
# Memory-Aware Model Suggestion
# =============================================================================
check_memory_and_suggest() {
    echo "Step: Memory Check"

    # Query available GPU memory via rocm-smi JSON output
    local vram_json
    vram_json=$(rocm-smi --showmeminfo vram --json 2>/dev/null) || {
        echo "Warning: Could not query GPU memory via rocm-smi, skipping memory check" >&2
        return 0
    }

    # Parse available VRAM (Total VRAM in bytes from JSON)
    # rocm-smi JSON format may use "card0" or similar keys with various key names
    local available_bytes
    available_bytes=$(echo "${vram_json}" | python -c "
import sys, json
data = json.load(sys.stdin)
# Navigate the JSON structure to find VRAM total
for key in data:
    card = data[key]
    if isinstance(card, dict):
        # Try common key patterns for total VRAM
        for mem_key in ('VRAM Total', 'vram_total', 'VRAM_Total', 'Total'):
            if mem_key in card:
                print(card[mem_key])
                sys.exit(0)
        # Try nested structure
        if 'vram' in card and isinstance(card['vram'], dict):
            for mem_key in ('total', 'Total', 'VRAM Total'):
                if mem_key in card['vram']:
                    print(card['vram'][mem_key])
                    sys.exit(0)
# Fallback: try top-level keys
for key in ('VRAM Total', 'vram_total', 'VRAM_Total'):
    if key in data:
        print(data[key])
        sys.exit(0)
print(0)
" 2>/dev/null) || {
        echo "Warning: Could not parse GPU memory info, skipping memory check" >&2
        return 0
    }

    # Convert bytes to GB (rocm-smi reports in bytes)
    local available_gb
    available_gb=$(python -c "
val = int(${available_bytes})
# If value is very large, it's in bytes - convert to GB
if val > 1024:
    print('{:.1f}'.format(val / (1024**3)))
else:
    # Already in GB or a small number
    print('{:.1f}'.format(float(val)))
" 2>/dev/null) || {
        echo "Warning: Could not convert memory value, skipping memory check" >&2
        return 0
    }

    # Look up memory requirement for the specified model
    local required_gb=""
    for entry in "${SUPPORTED_MODELS[@]}"; do
        local entry_model entry_mem
        entry_model=$(echo "${entry}" | cut -d: -f1)
        entry_mem=$(echo "${entry}" | cut -d: -f2)
        if [[ "${entry_model}" == "${MODEL}" ]]; then
            required_gb="${entry_mem}"
            break
        fi
    done

    if [[ -z "${required_gb}" ]]; then
        echo "Warning: Could not find memory requirement for model '${MODEL}'" >&2
        return 0
    fi

    # Check if model fits in available memory
    local fits
    fits=$(python -c "print(1 if float(${available_gb}) >= float(${required_gb}) else 0)" 2>/dev/null)

    if [[ "${fits}" == "1" ]]; then
        echo "  Model '${MODEL}' requires ~${required_gb} GB, available: ~${available_gb} GB — OK"
        return 0
    fi

    # Model does not fit — report error
    echo "ERROR: Insufficient GPU memory for ${MODEL}" >&2
    echo "  Required: ~${required_gb} GB" >&2
    echo "  Available: ~${available_gb} GB" >&2

    # Find the largest model that fits (SUPPORTED_MODELS is ordered largest to smallest)
    local best_model=""
    local best_mem=0
    for entry in "${SUPPORTED_MODELS[@]}"; do
        local entry_model entry_mem
        entry_model=$(echo "${entry}" | cut -d: -f1)
        entry_mem=$(echo "${entry}" | cut -d: -f2)

        local entry_fits
        entry_fits=$(python -c "print(1 if float(${available_gb}) >= float(${entry_mem}) else 0)" 2>/dev/null)

        if [[ "${entry_fits}" == "1" ]]; then
            local is_larger
            is_larger=$(python -c "print(1 if float(${entry_mem}) > float(${best_mem}) else 0)" 2>/dev/null)
            if [[ "${is_larger}" == "1" ]]; then
                best_model="${entry_model}"
                best_mem="${entry_mem}"
            fi
        fi
    done

    if [[ -n "${best_model}" ]]; then
        echo "SUGGESTION: The largest model that fits your GPU is ${best_model} (~${best_mem} GB required)." >&2
        echo "  Run: ./setup.sh --model ${best_model}" >&2
    else
        echo "No supported model can run with the available GPU memory." >&2
    fi

    exit "${EXIT_OOM}"
}

# =============================================================================
# Execute Installation Steps
# =============================================================================
install_pytorch
install_vllm
check_memory_and_suggest

# =============================================================================
# vLLM Server Launch with OOM Retry Logic
# =============================================================================
VLLM_LOG_FILE="/tmp/vllm_server_${PORT}.log"

# Check if a server is already running and healthy on the target port
check_server_health() {
    local host="$1"
    local port="$2"
    local response
    response=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time 5 \
        "http://${host}:${port}/v1/models" 2>/dev/null || echo "000")
    [[ "${response}" == "200" ]]
}

# Launch vLLM server as a background process
# Arguments: $1 = max_model_len to use for this attempt
start_vllm_process() {
    local max_len="$1"

    echo "  Launching vLLM server (max-model-len=${max_len})..."

    python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --max-model-len "${max_len}" \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --dtype "${DTYPE}" \
        --trust-remote-code \
        > "${VLLM_LOG_FILE}" 2>&1 &

    echo $!
}

# Wait for vLLM to become ready, checking for OOM or timeout
# Returns: 0 = ready, 1 = OOM detected, 2 = timeout
wait_for_vllm_ready() {
    local pid="$1"
    local timeout="$2"
    local elapsed=0
    local poll_interval=5

    while [[ ${elapsed} -lt ${timeout} ]]; do
        # Check if process is still running
        if ! kill -0 "${pid}" 2>/dev/null; then
            # Process died — check for OOM in logs
            if grep -qi "out of memory\|OutOfMemoryError\|CUDA out of memory\|torch.cuda.OutOfMemoryError" "${VLLM_LOG_FILE}" 2>/dev/null; then
                return 1  # OOM
            fi
            # Process died for another reason
            echo "Error: vLLM process exited unexpectedly" >&2
            echo "  Log tail:" >&2
            tail -20 "${VLLM_LOG_FILE}" >&2
            return 2  # Treat as timeout/failure
        fi

        # Check for OOM in logs while process is still running
        if grep -qi "out of memory\|OutOfMemoryError\|CUDA out of memory\|torch.cuda.OutOfMemoryError" "${VLLM_LOG_FILE}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
            wait "${pid}" 2>/dev/null || true
            return 1  # OOM
        fi

        # Check if server is ready via health endpoint
        if check_server_health "${HOST}" "${PORT}"; then
            return 0  # Ready
        fi

        # Check for ready log message (vLLM prints "Uvicorn running on" when ready)
        if grep -qi "Uvicorn running on\|Application startup complete" "${VLLM_LOG_FILE}" 2>/dev/null; then
            # Give it a moment to fully bind
            sleep 2
            if check_server_health "${HOST}" "${PORT}"; then
                return 0  # Ready
            fi
        fi

        sleep "${poll_interval}"
        elapsed=$((elapsed + poll_interval))
    done

    # Timeout reached
    return 2
}

launch_vllm_server() {
    echo "Step: vLLM Server Launch"

    # Check if server is already running and healthy on the target port
    if check_server_health "${HOST}" "${PORT}"; then
        echo "  Server already running and healthy on ${HOST}:${PORT}, skipping launch"
        echo ""
        echo "  Endpoint: http://${HOST}:${PORT}/v1/chat/completions"
        return 0
    fi

    # --- First attempt: launch with configured MAX_MODEL_LEN ---
    local pid
    pid=$(start_vllm_process "${MAX_MODEL_LEN}")
    echo "  vLLM PID: ${pid}"
    echo "  Waiting up to ${MODEL_LOAD_TIMEOUT}s for model to load..."

    local wait_result
    wait_for_vllm_ready "${pid}" "${MODEL_LOAD_TIMEOUT}"
    wait_result=$?

    if [[ ${wait_result} -eq 0 ]]; then
        echo "  vLLM server is ready!"
        echo ""
        echo "  Endpoint: http://${HOST}:${PORT}/v1/chat/completions"
        return 0
    fi

    if [[ ${wait_result} -eq 1 ]]; then
        echo "  OOM detected during model load with max-model-len=${MAX_MODEL_LEN}" >&2

        # --- OOM Retry: launch with reduced max-model-len ---
        echo "  Retrying with --max-model-len ${OOM_RETRY_MAX_MODEL_LEN}..."
        # Clear log for retry
        > "${VLLM_LOG_FILE}"

        pid=$(start_vllm_process "${OOM_RETRY_MAX_MODEL_LEN}")
        echo "  vLLM PID (retry): ${pid}"
        echo "  Waiting up to ${MODEL_LOAD_TIMEOUT}s for model to load..."

        wait_for_vllm_ready "${pid}" "${MODEL_LOAD_TIMEOUT}"
        wait_result=$?

        if [[ ${wait_result} -eq 0 ]]; then
            echo "  vLLM server is ready (with reduced context length)!"
            echo ""
            echo "  Endpoint: http://${HOST}:${PORT}/v1/chat/completions"
            return 0
        fi

        if [[ ${wait_result} -eq 1 ]]; then
            echo "Error: OOM on retry with max-model-len=${OOM_RETRY_MAX_MODEL_LEN}" >&2
            echo "  Model '${MODEL}' exceeds available GPU memory" >&2
            exit "${EXIT_OOM}"
        fi

        # Retry timed out
        kill "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
        echo "Error: Timeout on retry — model did not load within ${MODEL_LOAD_TIMEOUT}s" >&2
        exit "${EXIT_TIMEOUT}"
    fi

    # wait_result == 2: timeout on first attempt
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
    echo "Error: Model load timed out after ${MODEL_LOAD_TIMEOUT}s" >&2
    exit "${EXIT_TIMEOUT}"
}

# =============================================================================
# Execute Server Launch and Health Check
# =============================================================================
launch_vllm_server

# Run health check and print final summary
if run_health_check; then
    print_summary "success"
    exit "${EXIT_SUCCESS}"
else
    print_summary "failure"
    exit "${EXIT_HEALTH_CHECK_FAILURE}"
fi
