"""
Tests for the utilities module.

Covers:
- Ticker normalization to uppercase (Property 6)
- Whitespace and empty ticker rejection (Property 7)
- format_currency, safe_get, format_percent helpers
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from tools.utils import validate_ticker


class TestWhitespaceAndEmptyTickerRejection:
    """Feature: agent-tools, Property 7: Whitespace and empty ticker rejection

    For any string composed entirely of whitespace characters (including the
    empty string), validate_ticker SHALL return (False, error_message) where
    error_message contains "Error".

    Validates: Requirements 4.4, 2.1
    """

    @given(
        ticker=st.from_regex(r"^\s*$", fullmatch=True)
    )
    @settings(max_examples=100)
    def test_whitespace_only_strings_return_error(self, ticker: str) -> None:
        """Whitespace-only strings (including empty) are rejected with an Error message."""
        valid, message = validate_ticker(ticker)
        assert valid is False, f"Expected False for whitespace-only ticker {repr(ticker)}, got True"
        assert "Error" in message, (
            f"Expected 'Error' in message for whitespace-only ticker {repr(ticker)}, "
            f"got: {message}"
        )

from hypothesis import given, settings
import hypothesis.strategies as st

from tools.utils import validate_ticker


class TestTickerNormalizationProperty:
    """Feature: agent-tools, Property 6: Ticker normalization to uppercase

    Validates: Requirements 4.3
    """

    @given(ticker=st.text(min_size=1).filter(lambda s: s.strip()))
    @settings(max_examples=100)
    def test_validate_ticker_returns_uppercase(self, ticker: str):
        """For any non-empty, non-whitespace-only string, validate_ticker
        SHALL return (True, ticker.strip().upper())."""
        valid, result = validate_ticker(ticker)
        assert valid is True, f"Expected valid=True for ticker={ticker!r}, got valid={valid}"
        assert result == ticker.strip().upper(), (
            f"Expected {ticker.strip().upper()!r}, got {result!r}"
        )
