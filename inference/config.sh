#!/usr/bin/env bash
# ============================================================================
# FinAgent Inference Configuration
# Pinned versions and default parameters for reproducible deployment
# ============================================================================

# --- Pinned Dependency Versions ---
readonly VLLM_VERSION="0.6.3"
readonly TORCH_VERSION="2.4.0+rocm6.2"
readonly ROCM_VERSION="6.2"

# --- Default Server Parameters ---
# These can be overridden via CLI flags or environment variables
readonly DEFAULT_MODEL="${FINAGENT_MODEL:-Qwen/Qwen3-8B}"
readonly DEFAULT_HOST="${FINAGENT_HOST:-0.0.0.0}"
readonly DEFAULT_PORT="${FINAGENT_PORT:-8000}"
readonly DEFAULT_MAX_MODEL_LEN="${FINAGENT_MAX_LEN:-32768}"

# --- Supported Models and Memory Requirements ---
# Format: "HuggingFace_ID:min_memory_gb:context_length"
readonly SUPPORTED_MODELS=(
    "Qwen/Qwen3-14B:32:32768"
    "Qwen/Qwen3-8B:18:32768"
    "Qwen/Qwen3-4B:10:32768"
    "Qwen/Qwen3-1.7B:5:32768"
)

# --- Supported Model Names (for validation) ---
readonly SUPPORTED_MODEL_NAMES=(
    "Qwen/Qwen3-14B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-1.7B"
)

# --- vLLM Server Launch Parameters ---
readonly TENSOR_PARALLEL_SIZE=1
readonly GPU_MEMORY_UTILIZATION="0.90"
readonly MAX_NUM_SEQS=8
readonly DTYPE="auto"

# --- Timeouts ---
readonly MODEL_LOAD_TIMEOUT=300  # seconds
readonly HEALTH_CHECK_TIMEOUT=30 # seconds
readonly OOM_RETRY_MAX_MODEL_LEN=4096

# --- Exit Codes ---
readonly EXIT_SUCCESS=0
readonly EXIT_GENERAL_FAILURE=1
readonly EXIT_GPU_NOT_DETECTED=2
readonly EXIT_OOM=3
readonly EXIT_HEALTH_CHECK_FAILURE=4
readonly EXIT_TIMEOUT=5
