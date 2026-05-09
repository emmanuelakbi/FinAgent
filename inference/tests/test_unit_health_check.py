"""Unit tests for health check success and failure paths.

Tests the Python parse_health_response function from validation.py
and verifies the health_check.sh script behavior for connection errors
and timeouts.

**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**
"""

import os
import subprocess

import pytest

from inference_validation import parse_health_response


# ============================================================================
# Tests for parse_health_response: success path
# ============================================================================


class TestParseHealthResponseSuccess:
    """Test parse_health_response with valid 200 responses."""

    def test_valid_200_response_returns_healthy(self, health_check_success_response):
        """HTTP 200 with non-empty content returns ('healthy', content).

        Validates: Requirement 8.2
        """
        status, message = parse_health_response(200, health_check_success_response)

        assert status == "healthy"
        assert message == "Hello!"

    def test_valid_200_with_long_content(self):
        """HTTP 200 with longer content still returns healthy.

        Validates: Requirement 8.2
        """
        body = {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "Qwen/Qwen3-8B",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
            },
        }

        status, message = parse_health_response(200, body)

        assert status == "healthy"
        assert message == "Hello! How can I help you today?"


# ============================================================================
# Tests for parse_health_response: empty content failure
# ============================================================================


class TestParseHealthResponseEmptyContent:
    """Test parse_health_response with 200 but empty content."""

    def test_200_with_empty_content_returns_unhealthy(
        self, health_check_empty_response
    ):
        """HTTP 200 with empty choices[0].message.content returns unhealthy.

        Validates: Requirement 8.4
        """
        status, message = parse_health_response(200, health_check_empty_response)

        assert status == "unhealthy"
        assert message == "Empty response from model"

    def test_200_with_no_choices_returns_unhealthy(self):
        """HTTP 200 with empty choices array returns unhealthy.

        Validates: Requirement 8.4
        """
        body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "Qwen/Qwen3-8B",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
        }

        status, message = parse_health_response(200, body)

        assert status == "unhealthy"
        assert message == "Empty response from model"

    def test_200_with_missing_choices_key_returns_unhealthy(self):
        """HTTP 200 with no choices key at all returns unhealthy.

        Validates: Requirement 8.4
        """
        body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "Qwen/Qwen3-8B",
        }

        status, message = parse_health_response(200, body)

        assert status == "unhealthy"
        assert message == "Empty response from model"


# ============================================================================
# Tests for parse_health_response: non-200 status codes
# ============================================================================


class TestParseHealthResponseNon200:
    """Test parse_health_response with non-200 status codes."""

    def test_400_bad_request(self, http_error_responses):
        """HTTP 400 returns unhealthy with status code in message.

        Validates: Requirement 8.4
        """
        body = http_error_responses[400]
        status, message = parse_health_response(400, body)

        assert status == "unhealthy"
        assert "400" in message
        assert "Bad request" in message

    def test_404_not_found(self, http_error_responses):
        """HTTP 404 returns unhealthy with status code in message.

        Validates: Requirement 8.4
        """
        body = http_error_responses[404]
        status, message = parse_health_response(404, body)

        assert status == "unhealthy"
        assert "404" in message
        assert "Model not found" in message

    def test_500_internal_server_error(self, http_error_responses):
        """HTTP 500 returns unhealthy with status code in message.

        Validates: Requirement 8.4
        """
        body = http_error_responses[500]
        status, message = parse_health_response(500, body)

        assert status == "unhealthy"
        assert "500" in message
        assert "Internal server error" in message

    def test_503_service_unavailable(self, http_error_responses):
        """HTTP 503 returns unhealthy with status code in message.

        Validates: Requirement 8.4
        """
        body = http_error_responses[503]
        status, message = parse_health_response(503, body)

        assert status == "unhealthy"
        assert "503" in message
        assert "Service unavailable" in message


# ============================================================================
# Tests for health_check.sh: connection refused
# ============================================================================


HEALTH_CHECK_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "health_check.sh",
)


class TestHealthCheckScriptConnectionRefused:
    """Test health_check.sh against a port with nothing listening."""

    def test_connection_refused_exit_code(self):
        """health_check.sh exits with code 1 on connection refused.

        Validates: Requirement 8.5
        """
        result = subprocess.run(
            [HEALTH_CHECK_SCRIPT, "--host", "127.0.0.1", "--port", "19999", "--timeout", "5"],
            capture_output=True,
            text=True,
            timeout=15,
        )

        assert result.returncode == 1

    def test_connection_refused_error_message(self):
        """health_check.sh reports 'Connection refused' in stderr.

        Validates: Requirement 8.5
        """
        result = subprocess.run(
            [HEALTH_CHECK_SCRIPT, "--host", "127.0.0.1", "--port", "19999", "--timeout", "5"],
            capture_output=True,
            text=True,
            timeout=15,
        )

        assert "Connection refused" in result.stderr


# ============================================================================
# Tests for health_check.sh: timeout
# ============================================================================


class TestHealthCheckScriptTimeout:
    """Test health_check.sh timeout behavior.

    Note: On some platforms (e.g., macOS), non-routable IPs may trigger
    connection refused (exit 1) rather than timeout (exit 2). The test
    validates that the script handles the failure appropriately in either case.
    """

    def test_timeout_exit_code_nonzero(self):
        """health_check.sh exits with non-zero code on unreachable host.

        Uses a non-routable IP (10.255.255.1) with a short timeout.
        Depending on platform, curl may return exit 7 (connection refused → exit 1)
        or exit 28 (timeout → exit 2). Both are valid failure modes.

        Validates: Requirement 8.3
        """
        result = subprocess.run(
            [HEALTH_CHECK_SCRIPT, "--host", "10.255.255.1", "--port", "8000", "--timeout", "2"],
            capture_output=True,
            text=True,
            timeout=15,
        )

        # Script must exit with non-zero (1 for connection error, 2 for timeout)
        assert result.returncode in (1, 2)

    def test_timeout_error_message(self):
        """health_check.sh reports an error message in stderr for unreachable host.

        Validates: Requirement 8.3
        """
        result = subprocess.run(
            [HEALTH_CHECK_SCRIPT, "--host", "10.255.255.1", "--port", "8000", "--timeout", "2"],
            capture_output=True,
            text=True,
            timeout=15,
        )

        # Should report either timeout or connection error
        assert (
            "Timeout" in result.stderr
            or "no response within" in result.stderr
            or "Connection refused" in result.stderr
            or "Health check failed" in result.stderr
        )


# ============================================================================
# Tests for request payload structure
# ============================================================================


class TestHealthCheckRequestPayload:
    """Verify the expected health check payload matches the spec."""

    def test_payload_has_required_fields(self, health_check_request_payload):
        """Health check payload contains all required fields per spec.

        The spec defines the payload as:
        {"model": "<model>", "messages": [{"role": "user", "content": "Say hello."}],
         "max_tokens": 50, "temperature": 0.1}

        Validates: Requirement 8.1
        """
        payload = health_check_request_payload

        assert "model" in payload
        assert "messages" in payload
        assert "max_tokens" in payload
        assert "temperature" in payload

    def test_payload_model_field_is_string(self, health_check_request_payload):
        """Payload model field is a string.

        Validates: Requirement 8.1
        """
        assert isinstance(health_check_request_payload["model"], str)

    def test_payload_messages_structure(self, health_check_request_payload):
        """Payload messages is a list with one user message 'Say hello.'.

        Validates: Requirement 8.1
        """
        messages = health_check_request_payload["messages"]

        assert isinstance(messages, list)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Say hello."

    def test_payload_max_tokens_is_50(self, health_check_request_payload):
        """Payload max_tokens is 50 per spec.

        Validates: Requirement 8.1
        """
        assert health_check_request_payload["max_tokens"] == 50

    def test_payload_temperature_is_0_1(self, health_check_request_payload):
        """Payload temperature is 0.1 per spec.

        Validates: Requirement 8.1
        """
        assert health_check_request_payload["temperature"] == 0.1

    def test_payload_matches_spec_exactly(self, health_check_request_payload, default_model):
        """Full payload matches the spec structure exactly.

        Validates: Requirement 8.1
        """
        expected = {
            "model": default_model,
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 50,
            "temperature": 0.1,
        }

        assert health_check_request_payload == expected
