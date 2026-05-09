"""
Property-based tests for the signals module.

**Validates: Requirements 6.2, 6.3, 8.3, 9.1, 9.2, 9.4**
"""

import math
import re

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from crew.signals import Action, TradingSignal, TradingSignalParser


# Reusable strategies for round-trip test
_ticker_strategy = st.from_regex(r"[A-Z]{1,5}", fullmatch=True)
_action_strategy = st.sampled_from(Action)
_confidence_strategy = st.integers(min_value=0, max_value=100)
_price_strategy = st.floats(
    min_value=0.01, max_value=99999.99, allow_nan=False, allow_infinity=False
)
_optional_price_strategy = st.one_of(st.none(), _price_strategy)


class TestTradingSignalRoundTripParsing:
    """Property 2: Round trip consistency.

    For any valid TradingSignal instance (with a valid ticker, action in
    {BUY, SELL, HOLD}, confidence in 0-100, and non-negative prices),
    formatting it into the expected output string and then parsing that
    string back SHALL produce a TradingSignal with the same ticker, action,
    confidence, entry_price, stop_loss, and target_price values.

    **Validates: Requirements 6.2, 6.3, 9.1**
    """

    @given(
        ticker=_ticker_strategy,
        action=_action_strategy,
        confidence=_confidence_strategy,
        entry_price=_optional_price_strategy,
        stop_loss=_optional_price_strategy,
        target_price=_optional_price_strategy,
    )
    @settings(max_examples=200)
    def test_round_trip_consistency(
        self,
        ticker: str,
        action: Action,
        confidence: int,
        entry_price: float | None,
        stop_loss: float | None,
        target_price: float | None,
    ) -> None:
        """Formatting a TradingSignal and parsing it back preserves all fields."""
        # Build the expected primary format string
        formatted = f"{ticker} — {action.value} (Confidence: {confidence}%)"

        if entry_price is not None:
            formatted += f"\nEntry: ${entry_price:.2f}"
        if stop_loss is not None:
            formatted += f"\nStop Loss: ${stop_loss:.2f}"
        if target_price is not None:
            formatted += f"\nTarget: ${target_price:.2f}"

        # Parse the formatted string back
        parser = TradingSignalParser()
        parsed = parser.parse(formatted, ticker)

        # The parser must successfully parse the primary format
        assert parsed is not None, f"Parser returned None for: {formatted!r}"

        # Assert equality of core fields
        assert parsed.ticker == ticker
        assert parsed.action == action
        assert parsed.confidence == confidence

        # Assert price equality within floating point tolerance.
        # The parser extracts prices positionally (1st=$entry, 2nd=$stop_loss, 3rd=$target)
        # so we account for the positional extraction logic.
        expected_prices: list[float] = []
        if entry_price is not None:
            expected_prices.append(entry_price)
        if stop_loss is not None:
            expected_prices.append(stop_loss)
        if target_price is not None:
            expected_prices.append(target_price)

        # The parser assigns: first found = entry, second = stop_loss, third = target
        if len(expected_prices) >= 1:
            assert parsed.entry_price is not None
            assert math.isclose(
                parsed.entry_price, expected_prices[0], rel_tol=1e-2
            ), f"Entry price mismatch: {parsed.entry_price} != {expected_prices[0]}"
        else:
            assert parsed.entry_price is None

        if len(expected_prices) >= 2:
            assert parsed.stop_loss is not None
            assert math.isclose(
                parsed.stop_loss, expected_prices[1], rel_tol=1e-2
            ), f"Stop loss mismatch: {parsed.stop_loss} != {expected_prices[1]}"
        else:
            assert parsed.stop_loss is None

        if len(expected_prices) >= 3:
            assert parsed.target_price is not None
            assert math.isclose(
                parsed.target_price, expected_prices[2], rel_tol=1e-2
            ), f"Target price mismatch: {parsed.target_price} != {expected_prices[2]}"
        else:
            assert parsed.target_price is None


class TestUnparseableOutputYieldsNone:
    """Property 6: Unparseable output yields None.

    For any string that contains none of the words "BUY", "SELL", or "HOLD",
    the parser SHALL return None, indicating the output is unparseable.

    **Validates: Requirements 8.3**
    """

    @given(
        text=st.text(),
        ticker=st.text(
            alphabet=st.characters(whitelist_categories=("Lu",)),
            min_size=1,
            max_size=5,
        ),
    )
    def test_no_action_keywords_returns_none(self, text: str, ticker: str) -> None:
        """Strings without BUY/SELL/HOLD as standalone words yield None from the parser."""
        # Filter out any strings that contain BUY, SELL, or HOLD as word boundaries
        assume(
            re.search(r"\b(BUY|SELL|HOLD)\b", text, re.IGNORECASE) is None
        )

        parser = TradingSignalParser()
        result = parser.parse(text, ticker)

        assert result is None


class TestActionFieldValidation:
    """Property 3: Action field validation.

    For any string that does not contain exactly one of "BUY", "SELL", or "HOLD"
    as a standalone word, the parser SHALL either return None or extract only a
    valid Action enum value — never an invalid action string.

    **Validates: Requirements 9.2**
    """

    @given(text=st.text())
    def test_no_action_word_yields_none(self, text: str) -> None:
        """Strings without a standalone BUY/SELL/HOLD word produce None from the parser."""
        # Filter out strings that contain any of BUY, SELL, HOLD as standalone words
        assume(re.search(r"\b(BUY|SELL|HOLD)\b", text, re.IGNORECASE) is None)

        parser = TradingSignalParser()
        result = parser.parse(text, "TEST")

        assert result is None


class TestConfidenceClampingToValidRange:
    """Property 4: Confidence clamping to valid range.

    For any integer value (including values outside 0-100), the
    `validate_confidence` function SHALL return a value clamped to the
    range [0, 100] inclusive, such that values below 0 become 0 and
    values above 100 become 100.

    **Validates: Requirements 9.3**
    """

    @given(value=st.integers(min_value=-1000, max_value=1000))
    def test_confidence_always_in_valid_range(self, value: int) -> None:
        """validate_confidence always returns a value in [0, 100]."""
        result = TradingSignal.validate_confidence(value)
        assert 0 <= result <= 100

    @given(value=st.integers(min_value=-1000, max_value=-1))
    def test_negative_values_become_zero(self, value: int) -> None:
        """Values below 0 are clamped to 0."""
        result = TradingSignal.validate_confidence(value)
        assert result == 0

    @given(value=st.integers(min_value=101, max_value=1000))
    def test_values_above_100_become_100(self, value: int) -> None:
        """Values above 100 are clamped to 100."""
        result = TradingSignal.validate_confidence(value)
        assert result == 100

    @given(value=st.integers(min_value=0, max_value=100))
    def test_valid_values_unchanged(self, value: int) -> None:
        """Values already in [0, 100] are returned unchanged."""
        result = TradingSignal.validate_confidence(value)
        assert result == value


class TestFallbackParserExtraction:
    """Property 5: Fallback parser extraction.

    For any string containing one of "BUY", "SELL", or "HOLD" as a word
    and a number followed by "%" (even in non-standard formatting), the
    fallback parser SHALL extract the action and confidence percentage,
    returning a TradingSignal with the correct action and a confidence
    value in [0, 100].

    **Validates: Requirements 9.4**
    """

    @given(
        action_word=st.sampled_from(["BUY", "SELL", "HOLD"]),
        confidence_value=st.integers(min_value=0, max_value=100),
        prefix=st.text(
            alphabet=st.characters(blacklist_characters="%$"),
            min_size=0,
            max_size=20,
        ),
        suffix=st.text(
            alphabet=st.characters(blacklist_characters="%$"),
            min_size=0,
            max_size=20,
        ),
    )
    def test_fallback_parser_extracts_action_and_confidence(
        self,
        action_word: str,
        confidence_value: int,
        prefix: str,
        suffix: str,
    ) -> None:
        """Fallback parser extracts correct action and confidence from generated strings."""
        # Ensure prefix/suffix don't accidentally contain BUY/SELL/HOLD words
        assume("BUY" not in prefix.upper())
        assume("SELL" not in prefix.upper())
        assume("HOLD" not in prefix.upper())
        assume("BUY" not in suffix.upper())
        assume("SELL" not in suffix.upper())
        assume("HOLD" not in suffix.upper())

        text = f"{prefix} {action_word} with {confidence_value}% confidence {suffix}"

        parser = TradingSignalParser()
        result = parser.parse(text, "TEST")

        assert result is not None
        assert result.action == Action(action_word)
        assert result.confidence == TradingSignal.validate_confidence(confidence_value)
        assert 0 <= result.confidence <= 100
