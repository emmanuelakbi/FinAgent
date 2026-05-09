"""Unit tests for TradingSignalParser (Task 3.2)."""

import pytest

from crew.signals import Action, TradingSignal, TradingSignalParser


@pytest.fixture
def parser():
    return TradingSignalParser()


class TestParsePrimary:
    """Tests for the primary pattern parsing."""

    def test_standard_format_buy(self, parser):
        raw = "AAPL — BUY (Confidence: 75%)\nEntry: $185.42\nStop Loss: $180.55\nTarget: $192.17"
        signal = parser.parse(raw, "AAPL")
        assert signal is not None
        assert signal.ticker == "AAPL"
        assert signal.action == Action.BUY
        assert signal.confidence == 75
        assert signal.entry_price == 185.42
        assert signal.stop_loss == 180.55
        assert signal.target_price == 192.17

    def test_standard_format_sell(self, parser):
        raw = "MSFT — SELL (Confidence: 30%)\nEntry: $420.00\nStop Loss: $430.00\nTarget: $400.00"
        signal = parser.parse(raw, "MSFT")
        assert signal is not None
        assert signal.ticker == "MSFT"
        assert signal.action == Action.SELL
        assert signal.confidence == 30

    def test_standard_format_hold(self, parser):
        raw = "GOOGL — HOLD (Confidence: 50%)"
        signal = parser.parse(raw, "GOOGL")
        assert signal is not None
        assert signal.ticker == "GOOGL"
        assert signal.action == Action.HOLD
        assert signal.confidence == 50

    def test_en_dash_separator(self, parser):
        raw = "AAPL – BUY (Confidence: 80%)"
        signal = parser.parse(raw, "AAPL")
        assert signal is not None
        assert signal.action == Action.BUY
        assert signal.confidence == 80

    def test_hyphen_separator(self, parser):
        raw = "AAPL - BUY (Confidence: 60%)"
        signal = parser.parse(raw, "AAPL")
        assert signal is not None
        assert signal.action == Action.BUY
        assert signal.confidence == 60

    def test_ticker_with_dot(self, parser):
        raw = "BRK.B — BUY (Confidence: 70%)"
        signal = parser.parse(raw, "BRK.B")
        assert signal is not None
        assert signal.ticker == "BRK.B"

    def test_ticker_with_hyphen(self, parser):
        raw = "BF-B — SELL (Confidence: 40%)"
        signal = parser.parse(raw, "BF-B")
        assert signal is not None
        assert signal.ticker == "BF-B"

    def test_confidence_clamped_above_100(self, parser):
        raw = "AAPL — BUY (Confidence: 150%)"
        signal = parser.parse(raw, "AAPL")
        assert signal is not None
        assert signal.confidence == 100

    def test_confidence_zero(self, parser):
        raw = "AAPL — HOLD (Confidence: 0%)"
        signal = parser.parse(raw, "AAPL")
        assert signal is not None
        assert signal.confidence == 0

    def test_case_insensitive_action(self, parser):
        raw = "AAPL — buy (Confidence: 65%)"
        signal = parser.parse(raw, "AAPL")
        assert signal is not None
        assert signal.action == Action.BUY


class TestParseFallback:
    """Tests for the fallback heuristic parsing."""

    def test_unstructured_buy_with_confidence(self, parser):
        raw = "Based on analysis, I recommend a BUY with 80% confidence."
        signal = parser.parse(raw, "AAPL")
        assert signal is not None
        assert signal.ticker == "AAPL"
        assert signal.action == Action.BUY
        assert signal.confidence == 80

    def test_unstructured_sell_no_confidence(self, parser):
        raw = "The recommendation is to SELL this stock."
        signal = parser.parse(raw, "AAPL")
        assert signal is not None
        assert signal.action == Action.SELL
        assert signal.confidence == 50  # Default

    def test_unstructured_hold(self, parser):
        raw = "I suggest we HOLD for now. Confidence is around 65%."
        signal = parser.parse(raw, "AAPL")
        assert signal is not None
        assert signal.action == Action.HOLD
        assert signal.confidence == 65

    def test_fallback_with_prices(self, parser):
        raw = "BUY recommendation. Entry at $150.00, stop at $145.00, target $160.00. Confidence: 72%"
        signal = parser.parse(raw, "TSLA")
        assert signal is not None
        assert signal.entry_price == 150.00
        assert signal.stop_loss == 145.00
        assert signal.target_price == 160.00

    def test_fallback_uses_provided_ticker(self, parser):
        raw = "The stock should be a BUY at 75% confidence."
        signal = parser.parse(raw, "NVDA")
        assert signal is not None
        assert signal.ticker == "NVDA"


class TestUnparseableOutput:
    """Tests for outputs that cannot be parsed."""

    def test_no_action_returns_none(self, parser):
        raw = "The market is volatile today. No clear direction."
        signal = parser.parse(raw, "AAPL")
        assert signal is None

    def test_empty_string_returns_none(self, parser):
        signal = parser.parse("", "AAPL")
        assert signal is None

    def test_random_text_returns_none(self, parser):
        raw = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        signal = parser.parse(raw, "AAPL")
        assert signal is None


class TestExtractPrices:
    """Tests for price extraction."""

    def test_three_prices(self, parser):
        raw = "Entry: $185.42\nStop Loss: $180.55\nTarget: $192.17"
        prices = parser._extract_prices(raw)
        assert prices["entry"] == 185.42
        assert prices["stop_loss"] == 180.55
        assert prices["target"] == 192.17

    def test_one_price(self, parser):
        raw = "Entry at $150.00"
        prices = parser._extract_prices(raw)
        assert prices["entry"] == 150.00
        assert prices["stop_loss"] is None
        assert prices["target"] is None

    def test_no_prices(self, parser):
        raw = "No prices mentioned here."
        prices = parser._extract_prices(raw)
        assert prices["entry"] is None
        assert prices["stop_loss"] is None
        assert prices["target"] is None

    def test_prices_with_commas(self, parser):
        raw = "Entry: $1,250.50\nStop: $1,200.00\nTarget: $1,300.75"
        prices = parser._extract_prices(raw)
        assert prices["entry"] == 1250.50
        assert prices["stop_loss"] == 1200.00
        assert prices["target"] == 1300.75


class TestExtractReasoning:
    """Tests for reasoning extraction."""

    def test_full_reasoning(self, parser):
        raw = (
            "Reasoning:\n"
            "- Market: Positive earnings surprise\n"
            "- Fundamental: Undervalued relative to peers\n"
            "- Technical: MACD bullish crossover\n"
            "- Risk: 1:2 risk-reward ratio"
        )
        reasoning = parser._extract_reasoning(raw)
        assert reasoning is not None
        assert reasoning["Market"] == "Positive earnings surprise"
        assert reasoning["Fundamental"] == "Undervalued relative to peers"
        assert reasoning["Technical"] == "MACD bullish crossover"
        assert reasoning["Risk"] == "1:2 risk-reward ratio"

    def test_partial_reasoning(self, parser):
        raw = "- Market: Good news\n- Technical: Bullish"
        reasoning = parser._extract_reasoning(raw)
        assert reasoning is not None
        assert len(reasoning) == 2
        assert "Market" in reasoning
        assert "Technical" in reasoning

    def test_no_reasoning(self, parser):
        raw = "Just a plain text output with no reasoning markers."
        reasoning = parser._extract_reasoning(raw)
        assert reasoning is None


class TestFullParsePrimaryWithReasoning:
    """Integration test for primary parse with all fields."""

    def test_complete_signal(self, parser):
        raw = (
            "AAPL — BUY (Confidence: 75%)\n"
            "Entry: $185.42\n"
            "Stop Loss: $180.55\n"
            "Target: $192.17\n"
            "Reasoning:\n"
            "- Market: Positive earnings surprise and unusual volume detected\n"
            "- Fundamental: Undervalued relative to peers with strong growth\n"
            "- Technical: RSI neutral, MACD bullish crossover, price near support\n"
            "- Risk: 1:2 risk-reward ratio with ATR-based stop at $180.55"
        )
        signal = parser.parse(raw, "AAPL")
        assert signal is not None
        assert signal.ticker == "AAPL"
        assert signal.action == Action.BUY
        assert signal.confidence == 75
        assert signal.entry_price == 185.42
        assert signal.stop_loss == 180.55
        assert signal.target_price == 192.17
        assert signal.reasoning is not None
        assert "Market" in signal.reasoning
        assert "Fundamental" in signal.reasoning
        assert "Technical" in signal.reasoning
        assert "Risk" in signal.reasoning
