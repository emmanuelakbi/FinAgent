"""Property-based tests for model name validation.

Property 2: Invalid model name rejection with informative error

For any string that is not in the supported model set, validate_model_name
SHALL reject the input and produce an error message that contains both the
invalid value provided and the complete list of supported model variants.

Validates: Requirements 9.1, 9.4
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from inference_validation import validate_model_name, SUPPORTED_MODELS


# --- Property-Based Test ---


@given(
    name=st.text().filter(lambda s: s not in SUPPORTED_MODELS)
)
@settings(max_examples=100)
def test_invalid_model_name_rejected_with_informative_error(name: str) -> None:
    """Property 2: Invalid model name rejection with informative error.

    For any string not in the supported model set, validate_model_name returns
    a rejection tuple where:
    - The first element is False (rejected)
    - The error message contains the invalid value
    - The error message contains all 4 supported model names

    **Validates: Requirements 9.1, 9.4**
    """
    is_valid, error_message = validate_model_name(name)

    # Must be rejected
    assert is_valid is False, (
        f"Expected rejection for '{name}', but got accepted"
    )

    # Error message must contain the invalid value
    assert name in error_message, (
        f"Error message should contain the invalid value '{name}', "
        f"but got: '{error_message}'"
    )

    # Error message must contain all 4 supported model names
    for supported_model in SUPPORTED_MODELS:
        assert supported_model in error_message, (
            f"Error message should contain supported model '{supported_model}', "
            f"but got: '{error_message}'"
        )


# --- Parametrized Unit Test: Valid model names accepted ---


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_valid_model_names_accepted(model_name: str) -> None:
    """Verify all 4 supported model names ARE accepted (return True).

    **Validates: Requirements 9.1, 9.4**
    """
    is_valid, error_message = validate_model_name(model_name)

    assert is_valid is True, (
        f"Expected model '{model_name}' to be accepted, but got rejected "
        f"with error: '{error_message}'"
    )
    assert error_message == "", (
        f"Expected empty error message for valid model '{model_name}', "
        f"but got: '{error_message}'"
    )
