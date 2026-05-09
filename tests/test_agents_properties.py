"""
Property-based tests for the agents module.

**Validates: Requirements 1.2**
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from crew.agents import create_llm
from crew.config import LLMConfig


# Strategy to generate valid URL-like strings for base_url.
# Each hostname segment must contain at least one letter so we cannot
# accidentally generate IPv4-lookalikes like "0.0.0.00" which httpx/pydantic
# validate strictly as IP addresses.
base_url_strategy = st.from_regex(
    r"https?://[a-z][a-z0-9]*(\.[a-z][a-z0-9]*)*(:[0-9]{1,5})?(/[a-z0-9]*)*",
    fullmatch=True,
)


class TestLLMBaseUrlPropagation:
    """Property 1: LLM base_url propagation.

    For any valid URL string provided as `base_url` in the LLMConfig,
    when the LLM is created via `create_llm`, the resulting ChatOpenAI
    instance SHALL have its `base_url` set to that exact string.

    **Validates: Requirements 1.2**
    """

    @given(base_url=base_url_strategy)
    @settings(max_examples=200, deadline=None)
    def test_base_url_propagates_to_chat_openai(self, base_url: str) -> None:
        """The base_url from LLMConfig is propagated to the ChatOpenAI instance."""
        config = LLMConfig(base_url=base_url)
        llm = create_llm(config)

        # ChatOpenAI stores base_url as openai_api_base
        assert llm.openai_api_base == base_url, (
            f"Expected openai_api_base to be {base_url!r}, "
            f"got {llm.openai_api_base!r}"
        )
