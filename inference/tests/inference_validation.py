"""Pure Python implementations of inference setup validation logic.

These functions mirror the Bash logic in setup.sh and health_check.sh,
serving as the testable reference implementation for property-based tests.

Requirements: 9.1, 9.4, 9.5, 8.4
"""

from __future__ import annotations

# --- Module-Level Constants ---

SUPPORTED_MODELS: list[str] = [
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-1.7B",
]

# Model memory requirements in GB, sorted largest first
MODEL_MEMORY_REQUIREMENTS: dict[str, int] = {
    "Qwen/Qwen3-14B": 32,
    "Qwen/Qwen3-8B": 18,
    "Qwen/Qwen3-4B": 10,
    "Qwen/Qwen3-1.7B": 5,
}


def validate_model_name(name: str) -> tuple[bool, str]:
    """Validate a model name against the supported set.

    Args:
        name: The model name string to validate.

    Returns:
        A tuple of (is_valid, error_message).
        If valid: (True, "")
        If invalid: (False, error_message) where error_message contains
        the invalid value and all supported model names.

    Validates: Requirements 9.1, 9.4
    """
    if name in SUPPORTED_MODELS:
        return (True, "")

    supported_list = ", ".join(SUPPORTED_MODELS)
    error_message = (
        f"Error: Invalid model '{name}'. "
        f"Supported models: {supported_list}"
    )
    return (False, error_message)


def suggest_model(available_memory_gb: float) -> str | None:
    """Suggest the largest model that fits within available GPU memory.

    Iterates through models sorted by memory requirement (largest first)
    and returns the first model whose minimum memory requirement is less
    than or equal to the available memory.

    Args:
        available_memory_gb: Available GPU memory in gigabytes.

    Returns:
        The HuggingFace model ID of the largest fitting model,
        or None if no model fits.

    Validates: Requirements 9.5
    """
    for model_name, min_memory in MODEL_MEMORY_REQUIREMENTS.items():
        if min_memory <= available_memory_gb:
            return model_name
    return None


def parse_health_response(status_code: int, body: dict) -> tuple[str, str]:
    """Parse an HTTP health check response into a status and message.

    Args:
        status_code: The HTTP status code from the health check request.
        body: The parsed JSON response body.

    Returns:
        A tuple of (status, message) where:
        - status is "healthy" or "unhealthy"
        - message is the response content on success, or error details on failure.

    Validates: Requirements 8.4
    """
    if status_code != 200:
        # Extract error details from the response body
        error_details = ""
        if isinstance(body, dict):
            error_obj = body.get("error", {})
            if isinstance(error_obj, dict):
                error_details = error_obj.get("message", str(body))
            else:
                error_details = str(body)
        else:
            error_details = str(body)
        return ("unhealthy", f"HTTP {status_code}: {error_details}")

    # Status code is 200 — check for valid content in choices
    try:
        choices = body.get("choices", [])
        if choices and len(choices) > 0:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if content:
                return ("healthy", content)
    except (AttributeError, IndexError, TypeError):
        pass

    return ("unhealthy", "Empty response from model")
