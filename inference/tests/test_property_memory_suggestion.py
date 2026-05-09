"""Property-based tests for memory-aware model suggestion.

**Validates: Requirements 9.5**

Tests that suggest_model selects the largest fitting model for any
available GPU memory value, and that the suggestion is monotonic
(more memory never suggests a smaller model).
"""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from inference_validation import suggest_model, MODEL_MEMORY_REQUIREMENTS


# --- Constants for test assertions ---

# Models sorted by memory requirement descending (largest first)
MODELS_SORTED_DESC = sorted(
    MODEL_MEMORY_REQUIREMENTS.items(), key=lambda x: x[1], reverse=True
)

# The smallest memory requirement across all models
MIN_MODEL_MEMORY = min(MODEL_MEMORY_REQUIREMENTS.values())


class TestSuggestModelLargestFitting:
    """Property: suggest_model returns the largest fitting model.

    **Validates: Requirements 9.5**
    """

    @given(
        available_memory_gb=st.floats(
            min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False
        )
    )
    @settings(max_examples=100)
    def test_suggest_model_returns_largest_fitting_model(self, available_memory_gb):
        """For any memory value, suggest_model returns the largest model that fits."""
        result = suggest_model(available_memory_gb)

        if available_memory_gb < MIN_MODEL_MEMORY:
            # No model fits — should return None
            assert result is None, (
                f"Expected None when available memory ({available_memory_gb} GB) "
                f"is below minimum requirement ({MIN_MODEL_MEMORY} GB), got {result}"
            )
        else:
            # At least one model fits — should return a model
            assert result is not None, (
                f"Expected a model suggestion when available memory "
                f"({available_memory_gb} GB) >= {MIN_MODEL_MEMORY} GB"
            )

            # The suggested model's requirement must fit within available memory
            suggested_memory = MODEL_MEMORY_REQUIREMENTS[result]
            assert suggested_memory <= available_memory_gb, (
                f"Suggested model {result} requires {suggested_memory} GB "
                f"but only {available_memory_gb} GB available"
            )

            # No larger model should also fit (i.e., it's the LARGEST fitting model)
            for model_name, min_memory in MODELS_SORTED_DESC:
                if min_memory <= available_memory_gb:
                    # This is the largest fitting model — it should be the suggestion
                    assert result == model_name, (
                        f"Expected largest fitting model {model_name} "
                        f"(requires {min_memory} GB) but got {result} "
                        f"(requires {MODEL_MEMORY_REQUIREMENTS[result]} GB) "
                        f"with {available_memory_gb} GB available"
                    )
                    break


class TestSuggestModelMonotonicity:
    """Property: more memory never suggests a smaller model.

    **Validates: Requirements 9.5**
    """

    @given(
        memory_a=st.floats(
            min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False
        ),
        memory_b=st.floats(
            min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100)
    def test_monotonicity_more_memory_never_suggests_smaller_model(
        self, memory_a, memory_b
    ):
        """If memory_a >= memory_b, suggest_model(memory_a) >= suggest_model(memory_b)."""
        # Ensure memory_a >= memory_b
        if memory_a < memory_b:
            memory_a, memory_b = memory_b, memory_a

        result_a = suggest_model(memory_a)
        result_b = suggest_model(memory_b)

        if result_b is None:
            # If smaller memory returns None, larger memory can return anything (None or a model)
            pass
        elif result_a is None:
            # Larger memory should never return None if smaller memory returns a model
            assert False, (
                f"Monotonicity violated: suggest_model({memory_b}) = {result_b} "
                f"but suggest_model({memory_a}) = None (memory_a >= memory_b)"
            )
        else:
            # Both return models — the one with more memory should suggest
            # a model with >= memory requirement
            req_a = MODEL_MEMORY_REQUIREMENTS[result_a]
            req_b = MODEL_MEMORY_REQUIREMENTS[result_b]
            assert req_a >= req_b, (
                f"Monotonicity violated: suggest_model({memory_a}) = {result_a} "
                f"(requires {req_a} GB) < suggest_model({memory_b}) = {result_b} "
                f"(requires {req_b} GB), but memory_a >= memory_b"
            )


class TestSuggestModelBoundaryValues:
    """Test exact boundary values for model suggestion.

    **Validates: Requirements 9.5**
    """

    @pytest.mark.parametrize(
        "available_memory_gb,expected_model",
        [
            (4.9, None),
            (5.0, "Qwen/Qwen3-1.7B"),
            (9.9, "Qwen/Qwen3-1.7B"),
            (10.0, "Qwen/Qwen3-4B"),
            (17.9, "Qwen/Qwen3-4B"),
            (18.0, "Qwen/Qwen3-8B"),
            (31.9, "Qwen/Qwen3-8B"),
            (32.0, "Qwen/Qwen3-14B"),
        ],
    )
    def test_boundary_values(self, available_memory_gb, expected_model):
        """Test that exact boundary values produce the correct model suggestion."""
        result = suggest_model(available_memory_gb)
        assert result == expected_model, (
            f"For {available_memory_gb} GB available memory, "
            f"expected {expected_model} but got {result}"
        )
