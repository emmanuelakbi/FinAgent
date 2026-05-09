"""Property-based tests for health check failure reporting.

Property 1: Health check reports failure for any non-200 status code.

For any HTTP status code that is not 200 (including all 4xx and 5xx codes),
when the API endpoint returns that status code with any error body, the health
check SHALL report failure including the exact status code and the error body
content, and exit with a non-zero exit indication.

**Validates: Requirements 8.4**
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from inference_validation import parse_health_response


# Strategy: generate HTTP error status codes (4xx and 5xx ranges)
error_status_codes = st.one_of(
    st.integers(min_value=400, max_value=499),
    st.integers(min_value=500, max_value=599),
)

# Strategy: generate non-empty error message strings
error_messages = st.text(min_size=1, max_size=200).filter(lambda s: s.strip())


@given(status_code=error_status_codes, error_message=error_messages)
@settings(max_examples=100)
def test_health_check_reports_failure_for_non_200_status(status_code, error_message):
    """Property 1: Health check reports failure for any non-200 status code.

    For any 4xx or 5xx status code with any error body, parse_health_response
    must return unhealthy status with the status code and error content in the message.

    **Validates: Requirements 8.4**
    """
    body = {"error": {"message": error_message}}

    status, message = parse_health_response(status_code, body)

    # Must report unhealthy status
    assert status == "unhealthy", (
        f"Expected 'unhealthy' for HTTP {status_code}, got '{status}'"
    )

    # Must include the exact status code in the message
    assert str(status_code) in message, (
        f"Expected status code '{status_code}' in message, got: {message}"
    )

    # Must include the error body content in the message
    assert error_message in message, (
        f"Expected error body '{error_message}' in message, got: {message}"
    )


@given(status_code=error_status_codes)
@settings(max_examples=100)
def test_health_check_failure_with_non_dict_body(status_code):
    """Property 1 (variant): Non-200 status with non-dict error body still reports failure.

    Even when the body structure doesn't follow the standard error format,
    the health check must still report unhealthy with the status code.

    **Validates: Requirements 8.4**
    """
    body = {"unexpected": "format"}

    status, message = parse_health_response(status_code, body)

    # Must report unhealthy status
    assert status == "unhealthy", (
        f"Expected 'unhealthy' for HTTP {status_code}, got '{status}'"
    )

    # Must include the exact status code in the message
    assert str(status_code) in message, (
        f"Expected status code '{status_code}' in message, got: {message}"
    )


def test_health_check_200_with_empty_content_returns_unhealthy():
    """HTTP 200 with empty content should return unhealthy.

    When the API returns 200 but the response body has empty content
    in choices, the health check should report unhealthy.

    **Validates: Requirements 8.4**
    """
    body = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "Qwen/Qwen3-8B",
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

    status, message = parse_health_response(200, body)

    assert status == "unhealthy", (
        f"Expected 'unhealthy' for HTTP 200 with empty content, got '{status}'"
    )
    assert "Empty response from model" in message, (
        f"Expected 'Empty response from model' in message, got: {message}"
    )
