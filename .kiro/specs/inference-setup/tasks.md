# Implementation Plan: Inference Setup

## Overview

This plan implements the inference infrastructure automation for FinAgent on AMD MI300X hardware. The implementation uses Bash for the setup and health check scripts, and Python (Hypothesis) for property-based and unit tests. Each task builds incrementally — starting with project structure and core validation logic, then the main setup script, health check, fallback model logic, and finally tests that verify correctness properties.

## Tasks

- [x] 1. Set up project structure and core interfaces
  - [x] 1.1 Create directory structure and configuration constants
    - Create `inference/` directory with `setup.sh`, `health_check.sh`, `tests/`, and `README.md`
    - Create `inference/config.sh` with pinned versions (vLLM 0.6.3, PyTorch 2.4.0+rocm6.2, ROCm 6.2), default parameters (model, host, port, max-model-len), and supported model list with memory requirements
    - Make scripts executable with proper shebang lines (`#!/usr/bin/env bash`)
    - _Requirements: 7.4, 9.1, 9.2_

  - [x] 1.2 Create Python test infrastructure
    - Create `inference/tests/conftest.py` with shared fixtures for model names, memory values, and HTTP responses
    - Create `inference/tests/requirements.txt` with `hypothesis>=6.0`, `pytest>=7.0`, `pytest-hypothesis`
    - Create `inference/tests/__init__.py`
    - _Requirements: 7.4_

- [x] 2. Implement setup script (`setup.sh`)
  - [x] 2.1 Implement CLI argument parsing and validation
    - Parse `--model`, `--host`, `--port`, `--max-model-len` flags with defaults from `config.sh`
    - Validate model name against supported set (Qwen3-14B, Qwen3-8B, Qwen3-4B, Qwen3-1.7B)
    - On invalid model: print error with invalid value + supported list, exit code 1
    - Support environment variable overrides (`FINAGENT_MODEL`, `FINAGENT_HOST`, `FINAGENT_PORT`, `FINAGENT_MAX_LEN`)
    - _Requirements: 9.1, 9.4, 4.5_

  - [x] 2.2 Implement GPU detection and ROCm installation
    - Check GPU presence via `rocm-smi`; exit code 2 if not detected
    - Install/verify ROCm 6.2 (kernel driver, runtime, rocm-smi); exit code 1 on failure
    - Implement idempotency: skip install if correct version already present
    - Log each step name to stdout; log errors to stderr
    - _Requirements: 2.1, 2.2, 2.5, 7.3, 7.6_

  - [x] 2.3 Implement PyTorch and vLLM installation
    - Install ROCm 6.2-compatible PyTorch wheel matching active Python version
    - Verify GPU access via `python -c "import torch; assert torch.cuda.is_available()"`
    - Install vLLM 0.6.3 with ROCm backend
    - Use `--no-reinstall` / version checks for idempotency
    - Exit code 1 on any installation failure with step name in stderr
    - _Requirements: 2.3, 2.4, 2.6, 7.3, 7.4_

  - [x] 2.4 Implement vLLM server launch with OOM retry logic
    - Launch vLLM with: `--tensor-parallel-size 1 --gpu-memory-utilization 0.90 --max-num-seqs 8 --dtype auto --trust-remote-code`
    - Check if server already running on target port; skip launch if healthy
    - Wait up to 300s for model load (watch for ready log message)
    - On OOM: retry with `--max-model-len 4096`; exit code 3 if retry also fails
    - On timeout (300s): kill process, exit code 5
    - _Requirements: 3.1, 3.2, 3.4, 3.6, 5.1, 5.4, 7.3_

  - [x] 2.5 Implement memory-aware model suggestion
    - Query available GPU memory via `rocm-smi --showmeminfo vram --json`
    - Compare against model memory requirements table
    - If specified model exceeds available memory: report shortage and suggest largest fitting model
    - If no model fits: report that no supported model can run
    - _Requirements: 9.5_

  - [x] 2.6 Wire setup script end-to-end with health check integration
    - Call `health_check.sh` after successful server launch
    - Map health check exit codes to setup exit codes (health fail → exit 4)
    - Print final success/failure summary with model name and endpoint URL
    - Exit code 0 only when health check passes
    - _Requirements: 7.1, 7.2, 7.5, 7.7, 8.1_

- [x] 3. Checkpoint - Verify setup script structure
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement health check script (`health_check.sh`)
  - [x] 4.1 Implement health check request and response validation
    - Parse `--host`, `--port`, `--timeout` flags (defaults: 0.0.0.0, 8000, 30)
    - Detect loaded model name from `/v1/models` endpoint
    - Send POST to `/v1/chat/completions` with test payload: `{"model": "<detected>", "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 50, "temperature": 0.1}`
    - Validate: HTTP 200, non-empty `choices[0].message.content`
    - Report latency in milliseconds on success, exit code 0
    - _Requirements: 8.1, 8.2_

  - [x] 4.2 Implement health check error handling
    - Connection refused (curl exit 7): print "Connection refused at {host}:{port}", exit 1
    - Network unreachable (curl exit 6/28): print "Network unreachable", exit 1
    - Timeout (30s via `--max-time`): print timeout message, exit 2
    - Non-200 response: print status code + response body, exit 3
    - Empty response body: print "Empty response from model", exit 4
    - _Requirements: 8.3, 8.4, 8.5_

- [x] 5. Implement fallback model documentation
  - [x] 5.1 Create README with fallback model memory requirements
    - Document all 4 supported models with HuggingFace IDs, minimum GPU memory (GB), and context lengths
    - Include usage examples for switching models via `--model` flag
    - Document the automatic memory-aware suggestion behavior
    - Include troubleshooting section for common OOM scenarios
    - _Requirements: 9.2, 9.3_

- [x] 6. Checkpoint - Verify all scripts functional
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement property-based tests (Hypothesis)
  - [x] 7.1 Extract testable Python functions from shell logic
    - Create `inference/tests/validation.py` with pure Python implementations of:
      - `validate_model_name(name: str) -> tuple[bool, str]` — validates against supported set
      - `suggest_model(available_memory_gb: float) -> str | None` — returns largest fitting model
      - `parse_health_response(status_code: int, body: dict) -> tuple[str, str]` — returns (status, message)
    - These mirror the Bash logic and serve as the testable reference implementation
    - _Requirements: 9.1, 9.4, 9.5, 8.4_

  - [x] 7.2 Write property test for health check failure reporting
    - **Property 1: Health check reports failure for any non-200 status code**
    - Generate random HTTP status codes (4xx: 400-499, 5xx: 500-599) and random error body strings
    - Assert: `parse_health_response` returns failure status, includes exact status code, includes error body content, and returns non-zero exit indication
    - Minimum 100 iterations
    - **Validates: Requirements 8.4**

  - [x] 7.3 Write property test for invalid model name rejection
    - **Property 2: Invalid model name rejection with informative error**
    - Generate arbitrary strings via `hypothesis.strategies.text()` filtered to exclude the 4 supported names
    - Assert: `validate_model_name` returns rejection, error contains the invalid value, error contains all 4 supported model names
    - Minimum 100 iterations
    - **Validates: Requirements 9.1, 9.4**

  - [x] 7.4 Write property test for memory-aware model suggestion
    - **Property 3: Memory-aware model suggestion selects largest fitting model**
    - Generate random float memory values (0.0–200.0 GB) via `hypothesis.strategies.floats`
    - Assert: if any model fits, the suggested model is the largest one whose min memory ≤ available; if no model fits, returns None
    - Verify monotonicity: more memory never suggests a smaller model
    - Minimum 100 iterations
    - **Validates: Requirements 9.5**

- [x] 8. Implement unit tests
  - [x] 8.1 Write unit tests for CLI argument parsing
    - Test each supported model name is accepted
    - Test default model is Qwen3-8B when no `--model` flag
    - Test environment variable override precedence
    - Test invalid port values rejected
    - _Requirements: 9.1, 4.5_

  - [x] 8.2 Write unit tests for health check success and failure paths
    - Mock HTTP 200 with valid response: verify success report with latency
    - Mock HTTP 200 with empty content: verify failure report
    - Mock connection refused: verify error message format
    - Mock timeout: verify timeout error message
    - Verify request payload structure matches spec
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 8.3 Write unit tests for exit codes
    - Verify exit code 0 on full success path
    - Verify exit code 2 when GPU not detected
    - Verify exit code 3 on OOM after retry
    - Verify exit code 4 on health check failure
    - Verify exit code 5 on timeout
    - _Requirements: 7.7_

- [x] 9. Implement integration test scripts
  - [x] 9.1 Write integration test for full setup on MI300X
    - Script that runs `setup.sh` on fresh instance and verifies completion < 30 min
    - Verify `rocm-smi` detects MI300X after setup
    - Verify `torch.cuda.is_available()` returns True
    - Verify health check passes after setup
    - _Requirements: 7.1, 7.2, 2.2, 2.4_

  - [x] 9.2 Write integration test for concurrent requests and latency
    - Send 1 request (1024 input tokens, 512 max output): verify < 30s
    - Send 5 concurrent requests: verify all complete < 60s
    - Verify TTFT < 2s on idle server
    - Verify response isolation (no content leakage between concurrent requests)
    - _Requirements: 6.1, 6.2, 6.3, 5.1, 5.2, 5.3_

  - [x] 9.3 Write integration test for fallback model deployment
    - Deploy Qwen3-4B via `--model Qwen/Qwen3-4B`
    - Verify same `/v1/chat/completions` endpoint works with identical schema
    - Verify response contains valid assistant message
    - _Requirements: 9.3_

- [x] 10. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- Integration tests require actual MI300X hardware and should be run manually
- The Python validation module (`validation.py`) mirrors Bash logic to enable property testing with Hypothesis
- All scripts use pinned versions for reproducibility (vLLM 0.6.3, PyTorch 2.4.0+rocm6.2, ROCm 6.2)

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1", "4.1", "5.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "4.2"] },
    { "id": 3, "tasks": ["2.4", "2.5"] },
    { "id": 4, "tasks": ["2.6", "7.1"] },
    { "id": 5, "tasks": ["7.2", "7.3", "7.4"] },
    { "id": 6, "tasks": ["8.1", "8.2", "8.3"] },
    { "id": 7, "tasks": ["9.1", "9.2", "9.3"] }
  ]
}
```
