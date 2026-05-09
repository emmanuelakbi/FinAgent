# FinAgent Inference Setup

Automated infrastructure for deploying a high-performance LLM inference endpoint on AMD MI300X hardware using vLLM with ROCm support.

## Overview

This module provisions and configures the full inference stack:

- ROCm 6.2 GPU compute platform
- PyTorch with ROCm acceleration
- vLLM inference server with OpenAI-compatible API
- Health check verification

## Quick Start

```bash
# Deploy with default model (Qwen3-8B)
./setup.sh

# Deploy a specific model
./setup.sh --model Qwen/Qwen3-14B

# Custom host/port
./setup.sh --host 0.0.0.0 --port 8000 --max-model-len 32768

# Run health check
./health_check.sh --host 0.0.0.0 --port 8000
```

## Supported Models

| Model      | HuggingFace ID    | Min GPU Memory | Context Length |
| ---------- | ----------------- | -------------- | -------------- |
| Qwen3-14B  | `Qwen/Qwen3-14B`  | ~32 GB         | 32768 tokens   |
| Qwen3-8B   | `Qwen/Qwen3-8B`   | ~18 GB         | 32768 tokens   |
| Qwen3-4B   | `Qwen/Qwen3-4B`   | ~10 GB         | 32768 tokens   |
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` | ~5 GB          | 32768 tokens   |

## Fallback Models

When MI300X credits run out or you need to deploy on hardware with less GPU memory, the setup script supports automatic fallback to smaller Qwen3 variants. All models expose the same OpenAI-compatible `/v1/chat/completions` endpoint with identical request/response schemas — no client code changes are needed when switching models.

### Model Details

#### Qwen3-14B (Primary — Large)

- **HuggingFace ID:** `Qwen/Qwen3-14B`
- **Minimum GPU Memory:** ~32 GB
- **Context Length:** 32768 tokens
- **Use case:** Best quality for complex financial analysis. Requires MI300X (192 GB) or equivalent high-memory GPU.

#### Qwen3-8B (Default)

- **HuggingFace ID:** `Qwen/Qwen3-8B`
- **Minimum GPU Memory:** ~18 GB
- **Context Length:** 32768 tokens
- **Use case:** Balanced quality and resource usage. Default model. Runs on GPUs with 24+ GB VRAM (e.g., A10G, RTX 4090, MI300X).

#### Qwen3-4B (Fallback — Medium)

- **HuggingFace ID:** `Qwen/Qwen3-4B`
- **Minimum GPU Memory:** ~10 GB
- **Context Length:** 32768 tokens
- **Use case:** Reduced quality but fits on mid-range GPUs with 12+ GB VRAM (e.g., RTX 3080, T4).

#### Qwen3-1.7B (Fallback — Small)

- **HuggingFace ID:** `Qwen/Qwen3-1.7B`
- **Minimum GPU Memory:** ~5 GB
- **Context Length:** 32768 tokens
- **Use case:** Minimal quality, suitable for development/testing on consumer GPUs with 6+ GB VRAM (e.g., RTX 3060, RTX 2080).

### Switching Models

You can switch models using the `--model` CLI flag or the `FINAGENT_MODEL` environment variable.

#### Via CLI flag

```bash
# Deploy the default model (Qwen3-8B)
./setup.sh

# Deploy the 14B model for maximum quality
./setup.sh --model Qwen/Qwen3-14B

# Deploy the 4B fallback for mid-range GPUs
./setup.sh --model Qwen/Qwen3-4B

# Deploy the 1.7B model for development/testing
./setup.sh --model Qwen/Qwen3-1.7B
```

#### Via environment variable

```bash
# Set model via environment variable
export FINAGENT_MODEL="Qwen/Qwen3-4B"
./setup.sh

# Environment variable with other overrides
export FINAGENT_MODEL="Qwen/Qwen3-1.7B"
export FINAGENT_PORT=8080
./setup.sh
```

#### Precedence

CLI flags take priority over environment variables. If both are set, the CLI flag wins:

```bash
# This deploys Qwen3-14B (CLI flag wins)
export FINAGENT_MODEL="Qwen/Qwen3-4B"
./setup.sh --model Qwen/Qwen3-14B
```

### Automatic Memory-Aware Model Suggestion

The setup script automatically detects available GPU memory and provides intelligent suggestions when the requested model won't fit.

#### How it works

1. The script queries available VRAM via `rocm-smi --showmeminfo vram --json`
2. It compares the available memory against the minimum requirement for the requested model
3. If the requested model exceeds available memory, the script:
   - Reports the memory shortage (available vs. required)
   - Suggests the **largest** supported model that fits within available memory
   - If no model fits (less than ~5 GB available), reports that no supported model can run

#### Example output

When requesting Qwen3-14B on a GPU with 20 GB available:

```text
ERROR: Insufficient GPU memory for Qwen/Qwen3-14B
  Required: ~32 GB
  Available: 20 GB

SUGGESTION: The largest model that fits your GPU is Qwen/Qwen3-8B (~18 GB required).
  Run: ./setup.sh --model Qwen/Qwen3-8B
```

When requesting Qwen3-8B on a GPU with 8 GB available:

```text
ERROR: Insufficient GPU memory for Qwen/Qwen3-8B
  Required: ~18 GB
  Available: 8 GB

SUGGESTION: The largest model that fits your GPU is Qwen/Qwen3-4B (~10 GB required).
  Run: ./setup.sh --model Qwen/Qwen3-4B
```

When no model fits (less than 5 GB available):

```text
ERROR: Insufficient GPU memory for Qwen/Qwen3-1.7B
  Required: ~5 GB
  Available: 3 GB

No supported model can run with the available GPU memory.
Minimum requirement: ~5 GB for Qwen/Qwen3-1.7B.
```

#### Memory requirement table (quick reference)

| Available VRAM | Largest Model That Fits |
| -------------- | ----------------------- |
| 32+ GB         | Qwen3-14B               |
| 18–31 GB       | Qwen3-8B                |
| 10–17 GB       | Qwen3-4B                |
| 5–9 GB         | Qwen3-1.7B              |
| < 5 GB         | None (insufficient)     |

## Configuration

Parameters can be set via CLI flags or environment variables:

| Parameter   | CLI Flag          | Environment Variable | Default         |
| ----------- | ----------------- | -------------------- | --------------- |
| Model       | `--model`         | `FINAGENT_MODEL`     | `Qwen/Qwen3-8B` |
| Host        | `--host`          | `FINAGENT_HOST`      | `0.0.0.0`       |
| Port        | `--port`          | `FINAGENT_PORT`      | `8000`          |
| Max Context | `--max-model-len` | `FINAGENT_MAX_LEN`   | `32768`         |

## Pinned Versions

| Component | Version       |
| --------- | ------------- |
| vLLM      | 0.6.3         |
| PyTorch   | 2.4.0+rocm6.2 |
| ROCm      | 6.2           |

## Exit Codes

| Code | Meaning                  |
| ---- | ------------------------ |
| 0    | Success                  |
| 1    | General failure          |
| 2    | GPU not detected         |
| 3    | Model load failure (OOM) |
| 4    | Health check failure     |
| 5    | Timeout                  |

## Troubleshooting

### Common OOM Scenarios

#### Scenario 1: Model fails to load (Exit Code 3)

**Symptom:** The setup script exits with code 3 and an "out of memory" error.

**Cause:** The selected model's memory footprint exceeds available GPU VRAM. This can happen when:

- Other processes are consuming GPU memory
- The GPU has less VRAM than expected
- The full context length (32768 tokens) requires more KV cache than available

**Solutions:**

```bash
# Option A: Switch to a smaller model
./setup.sh --model Qwen/Qwen3-4B

# Option B: Reduce context length to lower KV cache memory usage
./setup.sh --model Qwen/Qwen3-8B --max-model-len 4096

# Option C: Check what's using GPU memory and free it
rocm-smi  # View current GPU memory usage
kill <pid>  # Kill other GPU processes if safe to do so
./setup.sh  # Retry
```

#### Scenario 2: OOM during inference (after successful load)

**Symptom:** The server loads successfully but returns errors or crashes during inference with long prompts.

**Cause:** The KV cache grows with input length. Long prompts or many concurrent requests can exhaust remaining GPU memory.

**Solutions:**

```bash
# Reduce max context length
./setup.sh --model Qwen/Qwen3-8B --max-model-len 16384

# Or switch to a smaller model that leaves more headroom
./setup.sh --model Qwen/Qwen3-4B --max-model-len 32768
```

#### Scenario 3: OOM retry triggered automatically

**Symptom:** The setup script logs "Retrying with --max-model-len 4096" during startup.

**Cause:** The initial model load with full context length (32768) failed due to insufficient memory. The script automatically retries with a reduced context length of 4096 tokens.

**What to expect:**

- If the retry succeeds: the server runs with reduced context (4096 tokens max per request)
- If the retry also fails (exit code 3): the model is too large for the available GPU entirely

**Solutions if retry fails:**

```bash
# Switch to the next smaller model
# If Qwen3-14B failed, try Qwen3-8B:
./setup.sh --model Qwen/Qwen3-8B

# If Qwen3-8B failed, try Qwen3-4B:
./setup.sh --model Qwen/Qwen3-4B

# If Qwen3-4B failed, try Qwen3-1.7B:
./setup.sh --model Qwen/Qwen3-1.7B
```

#### Scenario 4: GPU not detected (Exit Code 2)

**Symptom:** The setup script exits with code 2 and "GPU not detected" error.

**Cause:** ROCm cannot see the GPU. This typically means:

- ROCm drivers are not installed
- The instance doesn't have a GPU attached
- Driver/kernel mismatch

**Solutions:**

```bash
# Verify GPU hardware is present
lspci | grep -i amd

# Check ROCm installation
rocm-smi

# If ROCm is missing, the setup script will attempt to install it
# Ensure you're on a supported OS (Ubuntu 22.04 recommended)
./setup.sh
```

### General Tips

- **Check available memory before deploying:** Run `rocm-smi --showmeminfo vram` to see current VRAM availability.
- **Kill stale processes:** If a previous vLLM instance is still running, it holds GPU memory. The setup script checks for this, but you can manually verify with `ps aux | grep vllm`.
- **Use the memory-aware suggestion:** If you're unsure which model fits, try deploying the largest one. The script will suggest the best alternative if it doesn't fit.
- **Reduce context length first:** Before switching to a smaller model, try reducing `--max-model-len`. A context of 4096–8192 tokens is sufficient for many agent tasks.

## Directory Structure

```text
inference/
├── config.sh          # Pinned versions and default parameters
├── setup.sh           # Main setup automation script
├── health_check.sh    # Endpoint health verification
├── tests/             # Property-based and unit tests
└── README.md          # This file
```

## API Endpoint

Once running, the server exposes an OpenAI-compatible API:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```
