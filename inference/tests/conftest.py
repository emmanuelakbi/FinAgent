"""Shared pytest fixtures for inference setup tests.

Provides fixtures for model names, memory values, and HTTP responses
used across property-based and unit tests.
"""

import os
import sys

# Ensure this directory is on sys.path so tests can do
# `from inference_validation import ...` regardless of whether pytest is
# invoked from this directory or from the project root (e.g., `pytest
# inference/tests/`).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import pytest
from hypothesis import settings


# Configure Hypothesis: minimum 100 iterations per property test
settings.register_profile("default", max_examples=100)
settings.register_profile("ci", max_examples=200)
settings.load_profile("default")


# --- Supported Models ---

SUPPORTED_MODELS = {
    "Qwen/Qwen3-14B": {"min_memory_gb": 32, "context_length": 32768},
    "Qwen/Qwen3-8B": {"min_memory_gb": 18, "context_length": 32768},
    "Qwen/Qwen3-4B": {"min_memory_gb": 10, "context_length": 32768},
    "Qwen/Qwen3-1.7B": {"min_memory_gb": 5, "context_length": 32768},
}

SUPPORTED_MODEL_NAMES = list(SUPPORTED_MODELS.keys())

DEFAULT_MODEL = "Qwen/Qwen3-8B"

# Models sorted by memory requirement descending (largest first)
MODELS_BY_MEMORY_DESC = sorted(
    SUPPORTED_MODELS.items(), key=lambda x: x[1]["min_memory_gb"], reverse=True
)


# --- Model Fixtures ---


@pytest.fixture
def supported_models():
    """Return the full supported models dictionary."""
    return SUPPORTED_MODELS.copy()


@pytest.fixture
def supported_model_names():
    """Return list of supported model name strings."""
    return SUPPORTED_MODEL_NAMES.copy()


@pytest.fixture
def default_model():
    """Return the default model name."""
    return DEFAULT_MODEL


@pytest.fixture
def models_by_memory_desc():
    """Return models sorted by memory requirement, largest first."""
    return MODELS_BY_MEMORY_DESC.copy()


# --- Memory Value Fixtures ---


@pytest.fixture
def model_memory_requirements():
    """Return mapping of model names to minimum GPU memory in GB."""
    return {name: info["min_memory_gb"] for name, info in SUPPORTED_MODELS.items()}


@pytest.fixture
def mi300x_memory_gb():
    """Return MI300X total GPU memory in GB."""
    return 192


# --- HTTP Response Fixtures ---


@pytest.fixture
def health_check_success_response():
    """Return a successful health check HTTP response body."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": DEFAULT_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
        },
    }


@pytest.fixture
def health_check_empty_response():
    """Return a health check response with empty content."""
    return {
        "id": "chatcmpl-test456",
        "object": "chat.completion",
        "created": 1700000000,
        "model": DEFAULT_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 0,
            "total_tokens": 10,
        },
    }


@pytest.fixture
def health_check_request_payload():
    """Return the expected health check request payload structure."""
    return {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 50,
        "temperature": 0.1,
    }


@pytest.fixture
def http_error_responses():
    """Return sample HTTP error responses for various status codes."""
    return {
        400: {"error": {"message": "Bad request", "type": "invalid_request_error"}},
        404: {"error": {"message": "Model not found", "type": "not_found_error"}},
        422: {
            "error": {
                "message": "Validation error",
                "type": "validation_error",
            }
        },
        500: {
            "error": {
                "message": "Internal server error",
                "type": "server_error",
            }
        },
        503: {
            "error": {
                "message": "Service unavailable",
                "type": "service_unavailable",
            }
        },
    }
