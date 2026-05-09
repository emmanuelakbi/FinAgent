"""Integration tests for fallback model deployment (Qwen3-4B).

Tests that the fallback model (Qwen/Qwen3-4B) can be deployed via setup.sh
and serves at the same /v1/chat/completions endpoint with the standard
OpenAI ChatCompletion schema.

These tests require MI300X hardware and a running inference server.
Run with: cd inference/tests && python -m pytest test_integration_fallback.py -v -m integration

**Validates: Requirements 9.3**
"""

import os
import subprocess
import shutil

import pytest
import requests


# ============================================================================
# Hardware detection — skip if MI300X not available
# ============================================================================


def _has_mi300x_hardware() -> bool:
    """Check if MI300X hardware is available via rocm-smi."""
    if not shutil.which("rocm-smi"):
        return False
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and "MI300X" in result.stdout
    except (subprocess.TimeoutExpired, OSError):
        return False


requires_mi300x = pytest.mark.skipif(
    not _has_mi300x_hardware(),
    reason="MI300X hardware not available",
)


# ============================================================================
# Fixtures
# ============================================================================

SETUP_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "setup.sh",
)


@pytest.fixture(scope="module")
def server_url():
    """Base URL for the vLLM inference server.

    Override via FINAGENT_TEST_SERVER_URL environment variable.
    Default: http://0.0.0.0:8000
    """
    return os.environ.get("FINAGENT_TEST_SERVER_URL", "http://0.0.0.0:8000")


@pytest.fixture(scope="module")
def chat_completions_url(server_url):
    """Full URL for the /v1/chat/completions endpoint."""
    return f"{server_url}/v1/chat/completions"


@pytest.fixture(scope="module")
def fallback_model_name():
    """The fallback model being tested."""
    return "Qwen/Qwen3-4B"


@pytest.fixture(scope="module")
def chat_request_payload(fallback_model_name):
    """Standard chat completion request payload for testing."""
    return {
        "model": fallback_model_name,
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 50,
        "temperature": 0.1,
    }


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
@requires_mi300x
class TestFallbackModelDeployment:
    """Integration tests for deploying Qwen3-4B as a fallback model."""

    def test_deploy_qwen3_4b_via_setup_script(self):
        """Run setup.sh --model Qwen/Qwen3-4B and verify exit code 0.

        Validates: Requirement 9.3 — fallback model can be deployed via
        the same setup script with the --model parameter.
        """
        result = subprocess.run(
            [SETUP_SCRIPT, "--model", "Qwen/Qwen3-4B"],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minutes max for full setup
        )

        assert result.returncode == 0, (
            f"setup.sh --model Qwen/Qwen3-4B failed with exit code "
            f"{result.returncode}.\nstdout: {result.stdout[-500:]}\n"
            f"stderr: {result.stderr[-500:]}"
        )

    def test_fallback_model_serves_at_same_endpoint(
        self, chat_completions_url, chat_request_payload
    ):
        """Send POST to /v1/chat/completions and verify HTTP 200.

        Validates: Requirement 9.3 — the fallback model serves at the same
        /v1/chat/completions endpoint using the same request schema.
        """
        response = requests.post(
            chat_completions_url,
            json=chat_request_payload,
            timeout=60,
        )

        assert response.status_code == 200, (
            f"Expected HTTP 200 from /v1/chat/completions, "
            f"got {response.status_code}.\nBody: {response.text[:500]}"
        )

    def test_fallback_model_response_has_valid_schema(
        self, chat_completions_url, chat_request_payload
    ):
        """Verify response conforms to OpenAI ChatCompletion schema.

        The response must contain:
        - id: string
        - object: "chat.completion"
        - choices: array with at least one entry containing
          message.role="assistant" and non-empty message.content
        - usage: object with prompt_tokens, completion_tokens, total_tokens

        Validates: Requirement 9.3 — same OpenAI ChatCompletion response
        schema as the primary model.
        """
        response = requests.post(
            chat_completions_url,
            json=chat_request_payload,
            timeout=60,
        )
        assert response.status_code == 200

        data = response.json()

        # Verify top-level fields
        assert "id" in data, "Response missing 'id' field"
        assert isinstance(data["id"], str), "Response 'id' must be a string"

        assert "object" in data, "Response missing 'object' field"
        assert data["object"] == "chat.completion", (
            f"Expected object='chat.completion', got '{data['object']}'"
        )

        # Verify choices array
        assert "choices" in data, "Response missing 'choices' field"
        assert isinstance(data["choices"], list), "'choices' must be an array"
        assert len(data["choices"]) >= 1, "'choices' must have at least one entry"

        choice = data["choices"][0]
        assert "message" in choice, "choices[0] missing 'message' field"
        assert choice["message"]["role"] == "assistant", (
            f"Expected message.role='assistant', got '{choice['message']['role']}'"
        )
        assert "content" in choice["message"], "message missing 'content' field"
        assert isinstance(choice["message"]["content"], str), (
            "message.content must be a string"
        )
        assert len(choice["message"]["content"]) > 0, (
            "message.content must be non-empty"
        )

        # Verify usage object
        assert "usage" in data, "Response missing 'usage' field"
        usage = data["usage"]
        assert "prompt_tokens" in usage, "usage missing 'prompt_tokens'"
        assert "completion_tokens" in usage, "usage missing 'completion_tokens'"
        assert "total_tokens" in usage, "usage missing 'total_tokens'"
        assert isinstance(usage["prompt_tokens"], int)
        assert isinstance(usage["completion_tokens"], int)
        assert isinstance(usage["total_tokens"], int)

    def test_fallback_model_response_has_assistant_message(
        self, chat_completions_url, chat_request_payload
    ):
        """Verify choices[0].message.content is a non-empty string.

        Validates: Requirement 9.3 — response contains a valid assistant
        message, confirming the fallback model generates output.
        """
        response = requests.post(
            chat_completions_url,
            json=chat_request_payload,
            timeout=60,
        )
        assert response.status_code == 200

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        assert isinstance(content, str), (
            f"Expected content to be a string, got {type(content).__name__}"
        )
        assert len(content) > 0, (
            "Expected non-empty assistant message content"
        )
