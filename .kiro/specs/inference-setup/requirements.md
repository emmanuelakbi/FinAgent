# Requirements Document

## Introduction

This module covers the inference infrastructure setup for FinAgent — an autonomous multi-agent financial research and trading signal system. The goal is to deploy a high-performance LLM inference endpoint on AMD MI300X hardware using vLLM with ROCm support, exposing an OpenAI-compatible API that CrewAI agents can consume. This is the foundational layer that all downstream agent modules depend on.

## Glossary

- **Inference_Server**: The vLLM process serving the Qwen3 model on AMD MI300X hardware with ROCm acceleration
- **API_Endpoint**: The HTTP endpoint exposed by the Inference_Server at `/v1/chat/completions` following OpenAI API format
- **ROCm_Stack**: AMD's open-source GPU compute platform (version 6.2) including drivers, runtime, and libraries
- **MI300X_Instance**: An AMD Instinct MI300X GPU instance provisioned on AMD Developer Cloud
- **Setup_Script**: A reproducible automation script that provisions and configures the full inference stack
- **Agent_Client**: A CrewAI or LangChain agent that sends chat completion requests to the API_Endpoint
- **Health_Check**: A verification request confirming the Inference_Server is operational and responding correctly

## Requirements

### Requirement 1: Provision AMD MI300X Instance

**User Story:** As a developer, I want to provision an MI300X instance on AMD Developer Cloud, so that I have GPU hardware available for LLM inference.

#### Acceptance Criteria

1. WHEN the Setup_Script is executed, THE Setup_Script SHALL output to stdout the steps to provision an MI300X_Instance on AMD Developer Cloud
2. WHEN the MI300X_Instance is provisioned, THE MI300X_Instance SHALL have SSH access enabled on port 22 and be reachable via SSH within 120 seconds of provisioning completion
3. IF the MI300X_Instance fails to provision within 300 seconds, THEN THE Setup_Script SHALL output an error message to stderr indicating the failure reason and the provisioning step at which the failure occurred

### Requirement 2: Install ROCm and PyTorch

**User Story:** As a developer, I want ROCm 6.2 and PyTorch (ROCm wheel) installed on the instance, so that the GPU is accessible for model inference.

#### Acceptance Criteria

1. WHEN the Setup_Script installs the ROCm_Stack, THE Setup_Script SHALL install ROCm version 6.2 including the kernel-mode driver, the user-space ROCm runtime libraries, and the `rocm-smi` command-line tool
2. WHEN the ROCm_Stack installation completes, THE MI300X_Instance SHALL report the MI300X GPU as detected when `rocm-smi` is executed within 60 seconds of installation completion
3. WHEN the Setup_Script installs PyTorch, THE Setup_Script SHALL install the ROCm 6.2-compatible PyTorch wheel matching the Python version present in the active environment
4. WHEN PyTorch is installed, THE MI300X_Instance SHALL confirm GPU availability via `torch.cuda.is_available()` returning True within 60 seconds of installation completion
5. IF the ROCm_Stack installation fails, THEN THE Setup_Script SHALL log the failure details to the standard error output stream and halt execution with a non-zero exit code
6. IF the PyTorch installation fails, THEN THE Setup_Script SHALL log the failure details to the standard error output stream and halt execution with a non-zero exit code

### Requirement 3: Deploy Qwen3 Model via vLLM

**User Story:** As a developer, I want Qwen3-8B (or 14B) deployed via vLLM with ROCm support, so that the model is ready to serve inference requests.

#### Acceptance Criteria

1. WHEN the Setup_Script launches the Inference_Server, THE Inference_Server SHALL load the Qwen3-8B model using vLLM with ROCm backend and complete model loading within 300 seconds
2. WHEN the Inference_Server starts successfully, THE Inference_Server SHALL expose an HTTP endpoint and log a ready message that includes the serving port number, confirming the model is loaded and accepting requests
3. WHILE the Inference_Server is running, THE Inference_Server SHALL keep the Qwen3-8B model loaded in MI300X GPU memory and respond to health-check requests within 5 seconds
4. IF the Qwen3-8B model fails to load due to insufficient GPU memory, THEN THE Setup_Script SHALL reattempt loading Qwen3-8B with a maximum context length of 4096 tokens, and if the reattempt also fails, SHALL exit with a non-zero exit code and an error message indicating the memory requirement exceeded available GPU memory
5. WHERE the Qwen3-14B model is selected, THE Inference_Server SHALL load Qwen3-14B instead of Qwen3-8B
6. IF the Inference_Server does not complete model loading within 300 seconds, THEN THE Setup_Script SHALL terminate the loading process, exit with a non-zero exit code, and produce an error message indicating a model load timeout

### Requirement 4: Expose OpenAI-Compatible API Endpoint

**User Story:** As a developer, I want the inference server to expose an OpenAI-compatible API, so that CrewAI and LangChain agents can connect using a standard `base_url` configuration.

#### Acceptance Criteria

1. WHILE the Inference_Server is running, THE API_Endpoint SHALL accept POST requests at `/v1/chat/completions`
2. WHILE the Inference_Server is running, THE API_Endpoint SHALL accept request bodies containing the required fields `model` (string) and `messages` (non-empty array of objects with `role` and `content`), and the optional fields `temperature` (float, 0.0 to 2.0, default 1.0) and `max_tokens` (integer, 1 to 32768)
3. WHEN a valid chat completion request is received with messages containing roles `system`, `user`, or `assistant`, THE API_Endpoint SHALL return a JSON response containing the fields `id` (string), `object` (value `chat.completion`), `choices` (array with at least one entry containing `message.role` and `message.content`), and `usage`
4. WHEN a valid chat completion request is received, THE API_Endpoint SHALL include token usage statistics (`prompt_tokens`, `completion_tokens`, `total_tokens`) as integers in the `usage` object of the response
5. WHILE the Inference_Server is running, THE API_Endpoint SHALL be accessible on a host and port specified via command-line arguments or environment variables (default `0.0.0.0:8000`)
6. IF a request is received with missing required fields (`model` or `messages`), invalid field types, or an empty `messages` array, THEN THE API_Endpoint SHALL return an HTTP 422 response with an error body containing a message indicating which field failed validation
7. IF a request is received with a `model` value that does not match the currently loaded model, THEN THE API_Endpoint SHALL return an HTTP 404 response with an error body indicating the requested model is not available

### Requirement 5: Handle Concurrent Agent Requests

**User Story:** As a developer, I want the inference server to handle concurrent requests from multiple agents, so that all 5 CrewAI agents can query the model simultaneously without failures.

#### Acceptance Criteria

1. WHILE the Inference_Server is running, THE Inference_Server SHALL process at least 5 concurrent chat completion requests without returning HTTP error status codes
2. WHEN 5 concurrent requests are submitted each containing distinct identifiable prompt content, THE Inference_Server SHALL return a response conforming to the OpenAI ChatCompletion response schema for all 5 requests within 60 seconds
3. WHILE processing concurrent requests, THE Inference_Server SHALL maintain response isolation such that each response contains only content generated from its own request prompt and no content from other concurrent requests
4. IF the Inference_Server reaches its maximum concurrent request capacity, THEN THE Inference_Server SHALL queue additional requests and process them in received order rather than rejecting them
5. IF a queued request has not begun processing within 120 seconds of submission, THEN THE Inference_Server SHALL return an error response indicating a timeout to the requesting Agent_Client

### Requirement 6: Meet Response Latency Target

**User Story:** As a developer, I want response latency under 30 seconds for typical agent queries, so that the multi-agent workflow remains interactive and responsive.

#### Acceptance Criteria

1. WHEN a single chat completion request with up to 1024 input tokens and 512 max output tokens is submitted while no other requests are being processed, THE Inference_Server SHALL return a complete response within 30 seconds measured from request receipt to final token delivered
2. WHEN 5 concurrent chat completion requests each with up to 1024 input tokens and 512 max output tokens are submitted, THE Inference_Server SHALL return all complete responses within 60 seconds measured from the first request receipt to the last final token delivered
3. WHILE the Inference_Server has no active inference requests being processed, WHEN a chat completion request arrives, THE Inference_Server SHALL emit the first output token within 2 seconds of request receipt
4. IF a chat completion request does not complete within 60 seconds, THEN THE Inference_Server SHALL terminate the request and return an error response indicating a timeout failure

### Requirement 7: Reproducible Setup via Script

**User Story:** As a developer, I want the entire setup reproducible via a single script or documented steps, so that the environment can be rebuilt quickly within the hackathon timeframe.

#### Acceptance Criteria

1. THE Setup_Script SHALL automate the full installation sequence from a fresh MI300X_Instance to a running Inference_Server
2. THE Setup_Script SHALL complete the full setup in under 30 minutes on a fresh MI300X_Instance
3. IF the Setup_Script is run on an already-configured MI300X_Instance, THEN THE Setup_Script SHALL complete without error and the Health_Check SHALL pass and the Inference_Server SHALL remain operational
4. THE Setup_Script SHALL pin all dependency versions (vLLM, PyTorch, ROCm) to ensure reproducible builds
5. WHEN the Setup_Script completes successfully, THE Setup_Script SHALL run a Health_Check that returns a valid chat completion response within 30 seconds
6. IF any installation step fails, THEN THE Setup_Script SHALL halt execution and output the name of the failed step along with the error details to stderr
7. WHEN the Setup_Script completes, THE Setup_Script SHALL exit with code 0 on success or a non-zero exit code on failure

### Requirement 8: Endpoint Health Verification

**User Story:** As a developer, I want a health check that confirms the endpoint responds correctly, so that I can validate the setup before connecting agents.

#### Acceptance Criteria

1. WHEN the Health_Check is executed, THE Health_Check SHALL send a test chat completion request to the API_Endpoint containing a single user message of up to 50 tokens with max_tokens set to 50
2. WHEN the API_Endpoint returns an HTTP 200 response containing a non-empty assistant message (at least 1 character) in the choices array, THE Health_Check SHALL report success with response latency in milliseconds and exit with code 0
3. IF the API_Endpoint does not respond within 30 seconds, THEN THE Health_Check SHALL report failure with a timeout error message and exit with a non-zero exit code
4. IF the API_Endpoint returns a non-200 status code, THEN THE Health_Check SHALL report failure with the status code and error body and exit with a non-zero exit code
5. IF the Health_Check cannot establish a connection to the API_Endpoint (connection refused or network unreachable), THEN THE Health_Check SHALL report failure with a connection error message and exit with a non-zero exit code

### Requirement 9: Fallback Model Configuration

**User Story:** As a developer, I want documented steps to switch to a smaller model if MI300X credits run out, so that development can continue on alternative hardware.

#### Acceptance Criteria

1. THE Setup_Script SHALL accept a model name parameter specifying which Qwen3 variant to deploy, with supported values being Qwen3-14B, Qwen3-8B, Qwen3-4B, and Qwen3-1.7B, defaulting to Qwen3-8B when no parameter is provided
2. THE Setup_Script SHALL document fallback options including Qwen3-4B and Qwen3-1.7B, specifying for each variant the minimum GPU memory required in GB
3. WHEN a fallback model is specified, THE Inference_Server SHALL serve the specified model at the same `/v1/chat/completions` endpoint using the same OpenAI ChatCompletion request and response schema as the primary model
4. IF an invalid model name is specified, THEN THE Setup_Script SHALL exit with an error message indicating the invalid value and listing the supported model variants
5. IF the specified model exceeds available GPU memory, THEN THE Setup_Script SHALL detect available GPU memory via runtime query, report the shortage, and suggest the largest supported model variant that fits within the detected available memory
