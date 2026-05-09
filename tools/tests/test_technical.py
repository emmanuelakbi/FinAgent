"""
Tests for the Technical Analyst tools.

Covers:
- Signal classification from indicator values (Property 13)
- Unit tests for get_price_history and calculate_indicators
"""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from tools.technical_analyst import calculate_indicators, get_price_history, cache


class TestSignalClassificationProperty:
    """
    Feature: agent-tools, Property 13: Signal classification from indicator values

    Property-based test verifying signal classification logic for RSI, MACD, and Bollinger Bands.

    **Validates: Requirements 12.2, 12.3, 12.4**
    """

    def setup_method(self):
        """Clear cache before each test example."""
        cache.clear()

    @settings(max_examples=100)
    @given(rsi_value=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    def test_rsi_signal_classification(self, rsi_value):
        """
        Property 13a: RSI Signal Classification

        For any RSI value in [0, 100]:
        - RSI < 30 → output contains "BUY"
        - RSI > 70 → output contains "SELL"
        - 30 ≤ RSI ≤ 70 → output contains "NEUTRAL"

        **Validates: Requirements 12.2**
        """
        cache.clear()

        closes = [100.0] * 60
        hist_df = pd.DataFrame({"Close": closes})

        rsi_series = pd.Series([rsi_value] * 60)
        # MACD: no crossover (both prev and curr same relationship)
        macd_df = pd.DataFrame({
            "MACD_12_26_9": [1.0] * 60,
            "MACDh_12_26_9": [0.5] * 60,
            "MACDs_12_26_9": [0.5] * 60,
        })
        bbands_df = pd.DataFrame({
            "BBU_20_2.0": [110.0] * 60,
            "BBM_20_2.0": [100.0] * 60,
            "BBL_20_2.0": [90.0] * 60,
        })

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist_df

        with patch("tools.technical_analyst.yfinance.Ticker", return_value=mock_ticker), \
             patch("tools.technical_analyst.ta.rsi", return_value=rsi_series), \
             patch("tools.technical_analyst.ta.macd", return_value=macd_df), \
             patch("tools.technical_analyst.ta.bbands", return_value=bbands_df):
            result = calculate_indicators("AAPL")

        # Extract the RSI line from the result
        rsi_line = [line for line in result.split("\n") if "RSI (14):" in line]
        assert len(rsi_line) == 1, f"Expected exactly one RSI line, got: {result}"
        rsi_line = rsi_line[0]

        if rsi_value < 30:
            assert "BUY" in rsi_line, f"RSI={rsi_value} should be BUY, got: {rsi_line}"
        elif rsi_value > 70:
            assert "SELL" in rsi_line, f"RSI={rsi_value} should be SELL, got: {rsi_line}"
        else:
            assert "NEUTRAL" in rsi_line, f"RSI={rsi_value} should be NEUTRAL, got: {rsi_line}"

    @settings(max_examples=100)
    @given(
        prev_macd=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        prev_signal=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        curr_macd=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        curr_signal=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    def test_macd_signal_classification(self, prev_macd, prev_signal, curr_macd, curr_signal):
        """
        Property 13b: MACD Signal Classification

        For any (prev_macd, prev_signal, curr_macd, curr_signal):
        - Bullish crossover (curr_macd > curr_signal AND prev_macd <= prev_signal) → "BUY"
        - Bearish crossover (curr_macd < curr_signal AND prev_macd >= prev_signal) → "SELL"
        - Otherwise → "NEUTRAL"

        **Validates: Requirements 12.3**
        """
        cache.clear()

        closes = [100.0] * 60
        hist_df = pd.DataFrame({"Close": closes})

        rsi_series = pd.Series([50.0] * 60)  # Neutral RSI
        # Set up MACD values: previous at index -2, current at index -1
        macd_values = [0.0] * 58 + [prev_macd, curr_macd]
        signal_values = [0.0] * 58 + [prev_signal, curr_signal]
        macd_df = pd.DataFrame({
            "MACD_12_26_9": macd_values,
            "MACDh_12_26_9": [0.0] * 60,
            "MACDs_12_26_9": signal_values,
        })
        bbands_df = pd.DataFrame({
            "BBU_20_2.0": [110.0] * 60,
            "BBM_20_2.0": [100.0] * 60,
            "BBL_20_2.0": [90.0] * 60,
        })

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist_df

        with patch("tools.technical_analyst.yfinance.Ticker", return_value=mock_ticker), \
             patch("tools.technical_analyst.ta.rsi", return_value=rsi_series), \
             patch("tools.technical_analyst.ta.macd", return_value=macd_df), \
             patch("tools.technical_analyst.ta.bbands", return_value=bbands_df):
            result = calculate_indicators("AAPL")

        # Extract the MACD line from the result
        macd_line = [line for line in result.split("\n") if "MACD:" in line]
        assert len(macd_line) == 1, f"Expected exactly one MACD line, got: {result}"
        macd_line = macd_line[0]

        # Determine expected signal
        is_bullish_crossover = curr_macd > curr_signal and prev_macd <= prev_signal
        is_bearish_crossover = curr_macd < curr_signal and prev_macd >= prev_signal

        if is_bullish_crossover:
            assert "BUY" in macd_line, f"MACD bullish crossover should be BUY, got: {macd_line}"
        elif is_bearish_crossover:
            assert "SELL" in macd_line, f"MACD bearish crossover should be SELL, got: {macd_line}"
        else:
            assert "NEUTRAL" in macd_line, f"MACD no crossover should be NEUTRAL, got: {macd_line}"

    @settings(max_examples=100)
    @given(
        close=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        lower_band=st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        band_width=st.floats(min_value=0.01, max_value=500.0, allow_nan=False, allow_infinity=False),
    )
    def test_bollinger_signal_classification(self, close, lower_band, band_width):
        """
        Property 13c: Bollinger Signal Classification

        For any (close, upper_band, lower_band) where lower_band < upper_band:
        - close < lower_band → "BUY"
        - close > upper_band → "SELL"
        - Otherwise → "NEUTRAL"

        **Validates: Requirements 12.4**
        """
        upper_band = lower_band + band_width  # Ensures upper > lower
        cache.clear()

        closes = [close] * 60
        hist_df = pd.DataFrame({"Close": closes})

        rsi_series = pd.Series([50.0] * 60)  # Neutral RSI
        macd_df = pd.DataFrame({
            "MACD_12_26_9": [1.0] * 60,
            "MACDh_12_26_9": [0.5] * 60,
            "MACDs_12_26_9": [0.5] * 60,
        })
        bbands_df = pd.DataFrame({
            "BBU_20_2.0": [upper_band] * 60,
            "BBM_20_2.0": [(upper_band + lower_band) / 2] * 60,
            "BBL_20_2.0": [lower_band] * 60,
        })

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist_df

        with patch("tools.technical_analyst.yfinance.Ticker", return_value=mock_ticker), \
             patch("tools.technical_analyst.ta.rsi", return_value=rsi_series), \
             patch("tools.technical_analyst.ta.macd", return_value=macd_df), \
             patch("tools.technical_analyst.ta.bbands", return_value=bbands_df):
            result = calculate_indicators("AAPL")

        # Extract the Bollinger line from the result
        bb_line = [line for line in result.split("\n") if "Bollinger:" in line]
        assert len(bb_line) == 1, f"Expected exactly one Bollinger line, got: {result}"
        bb_line = bb_line[0]

        if close < lower_band:
            assert "BUY" in bb_line, f"Close={close} < Lower={lower_band} should be BUY, got: {bb_line}"
        elif close > upper_band:
            assert "SELL" in bb_line, f"Close={close} > Upper={upper_band} should be SELL, got: {bb_line}"
        else:
            assert "NEUTRAL" in bb_line, f"Close={close} between bands should be NEUTRAL, got: {bb_line}"


class TestCalculateIndicatorsUnit:
    """Unit tests for calculate_indicators function."""

    def setup_method(self):
        """Clear cache before each test."""
        cache.clear()

    def _make_hist_df(self, closes):
        """Helper to create a history DataFrame from a list of close prices."""
        return pd.DataFrame({"Close": closes})

    def _make_indicator_mocks(self, rsi_value, macd_values, signal_values, bbu_values, bbl_values):
        """Helper to create mock return values for pandas-ta indicators.

        Args:
            rsi_value: Current RSI value (last element)
            macd_values: List of at least 2 MACD values [previous, current]
            signal_values: List of at least 2 signal values [previous, current]
            bbu_values: List of upper band values
            bbl_values: List of lower band values
        """
        rsi_series = pd.Series([rsi_value] * 60)
        macd_df = pd.DataFrame({
            "MACD_12_26_9": macd_values,
            "MACDh_12_26_9": [0.0] * len(macd_values),
            "MACDs_12_26_9": signal_values,
        })
        bbands_df = pd.DataFrame({
            "BBU_20_2.0": bbu_values,
            "BBM_20_2.0": [0.0] * len(bbu_values),
            "BBL_20_2.0": bbl_values,
        })
        return rsi_series, macd_df, bbands_df

    def test_all_neutral_signals(self):
        """calculate_indicators returns NEUTRAL for all indicators when values are in neutral range."""
        # RSI = 50 (neutral), no MACD crossover, price between bands
        closes = [100.0 + i * 0.1 for i in range(60)]  # Gradual uptrend
        hist_df = self._make_hist_df(closes)

        rsi_series = pd.Series([50.0] * 60)
        macd_df = pd.DataFrame({
            "MACD_12_26_9": [1.0] * 60,  # MACD above signal, no crossover (both prev and curr above)
            "MACDh_12_26_9": [0.5] * 60,
            "MACDs_12_26_9": [0.5] * 60,
        })
        bbands_df = pd.DataFrame({
            "BBU_20_2.0": [110.0] * 60,  # Upper band above close
            "BBM_20_2.0": [105.0] * 60,
            "BBL_20_2.0": [95.0] * 60,   # Lower band below close
        })

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist_df

        with patch("tools.technical_analyst.yfinance.Ticker", return_value=mock_ticker), \
             patch("tools.technical_analyst.ta.rsi", return_value=rsi_series), \
             patch("tools.technical_analyst.ta.macd", return_value=macd_df), \
             patch("tools.technical_analyst.ta.bbands", return_value=bbands_df):
            result = calculate_indicators("AAPL")

        assert "Technical Signals for AAPL:" in result
        assert "RSI (14): 50.0 → NEUTRAL (Neutral)" in result
        assert "→ NEUTRAL (Neutral)" in result
        # Bollinger should be NEUTRAL since close (105.9) is between 95 and 110
        assert "Bollinger:" in result

    def test_rsi_buy_signal_oversold(self):
        """calculate_indicators returns BUY for RSI < 30 (oversold)."""
        closes = [100.0] * 60
        hist_df = self._make_hist_df(closes)

        rsi_series = pd.Series([25.0] * 60)
        macd_df = pd.DataFrame({
            "MACD_12_26_9": [1.0] * 60,
            "MACDh_12_26_9": [0.5] * 60,
            "MACDs_12_26_9": [0.5] * 60,
        })
        bbands_df = pd.DataFrame({
            "BBU_20_2.0": [110.0] * 60,
            "BBM_20_2.0": [100.0] * 60,
            "BBL_20_2.0": [90.0] * 60,
        })

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist_df

        with patch("tools.technical_analyst.yfinance.Ticker", return_value=mock_ticker), \
             patch("tools.technical_analyst.ta.rsi", return_value=rsi_series), \
             patch("tools.technical_analyst.ta.macd", return_value=macd_df), \
             patch("tools.technical_analyst.ta.bbands", return_value=bbands_df):
            result = calculate_indicators("AAPL")

        assert "RSI (14): 25.0 → BUY (Oversold)" in result

    def test_rsi_sell_signal_overbought(self):
        """calculate_indicators returns SELL for RSI > 70 (overbought)."""
        closes = [100.0] * 60
        hist_df = self._make_hist_df(closes)

        rsi_series = pd.Series([75.0] * 60)
        macd_df = pd.DataFrame({
            "MACD_12_26_9": [1.0] * 60,
            "MACDh_12_26_9": [0.5] * 60,
            "MACDs_12_26_9": [0.5] * 60,
        })
        bbands_df = pd.DataFrame({
            "BBU_20_2.0": [110.0] * 60,
            "BBM_20_2.0": [100.0] * 60,
            "BBL_20_2.0": [90.0] * 60,
        })

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist_df

        with patch("tools.technical_analyst.yfinance.Ticker", return_value=mock_ticker), \
             patch("tools.technical_analyst.ta.rsi", return_value=rsi_series), \
             patch("tools.technical_analyst.ta.macd", return_value=macd_df), \
             patch("tools.technical_analyst.ta.bbands", return_value=bbands_df):
            result = calculate_indicators("AAPL")

        assert "RSI (14): 75.0 → SELL (Overbought)" in result

    def test_macd_buy_signal_bullish_crossover(self):
        """calculate_indicators returns BUY for MACD bullish crossover."""
        closes = [100.0] * 60
        hist_df = self._make_hist_df(closes)

        rsi_series = pd.Series([50.0] * 60)
        # Bullish crossover: previous MACD <= previous signal, current MACD > current signal
        macd_values = [0.5] * 58 + [0.8, 1.2]   # previous=0.8, current=1.2
        signal_values = [0.5] * 58 + [0.9, 0.9]  # previous=0.9, current=0.9
        # prev MACD (0.8) <= prev signal (0.9) AND curr MACD (1.2) > curr signal (0.9) → BUY
        macd_df = pd.DataFrame({
            "MACD_12_26_9": macd_values,
            "MACDh_12_26_9": [0.0] * 60,
            "MACDs_12_26_9": signal_values,
        })
        bbands_df = pd.DataFrame({
            "BBU_20_2.0": [110.0] * 60,
            "BBM_20_2.0": [100.0] * 60,
            "BBL_20_2.0": [90.0] * 60,
        })

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist_df

        with patch("tools.technical_analyst.yfinance.Ticker", return_value=mock_ticker), \
             patch("tools.technical_analyst.ta.rsi", return_value=rsi_series), \
             patch("tools.technical_analyst.ta.macd", return_value=macd_df), \
             patch("tools.technical_analyst.ta.bbands", return_value=bbands_df):
            result = calculate_indicators("AAPL")

        assert "→ BUY (Bullish Crossover)" in result

    def test_macd_sell_signal_bearish_crossover(self):
        """calculate_indicators returns SELL for MACD bearish crossover."""
        closes = [100.0] * 60
        hist_df = self._make_hist_df(closes)

        rsi_series = pd.Series([50.0] * 60)
        # Bearish crossover: previous MACD >= previous signal, current MACD < current signal
        macd_values = [1.0] * 58 + [1.0, 0.7]   # previous=1.0, current=0.7
        signal_values = [0.5] * 58 + [0.9, 0.9]  # previous=0.9, current=0.9
        # prev MACD (1.0) >= prev signal (0.9) AND curr MACD (0.7) < curr signal (0.9) → SELL
        macd_df = pd.DataFrame({
            "MACD_12_26_9": macd_values,
            "MACDh_12_26_9": [0.0] * 60,
            "MACDs_12_26_9": signal_values,
        })
        bbands_df = pd.DataFrame({
            "BBU_20_2.0": [110.0] * 60,
            "BBM_20_2.0": [100.0] * 60,
            "BBL_20_2.0": [90.0] * 60,
        })

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist_df

        with patch("tools.technical_analyst.yfinance.Ticker", return_value=mock_ticker), \
             patch("tools.technical_analyst.ta.rsi", return_value=rsi_series), \
             patch("tools.technical_analyst.ta.macd", return_value=macd_df), \
             patch("tools.technical_analyst.ta.bbands", return_value=bbands_df):
            result = calculate_indicators("AAPL")

        assert "→ SELL (Bearish Crossover)" in result

    def test_bollinger_buy_signal_below_lower(self):
        """calculate_indicators returns BUY when close < lower Bollinger Band."""
        closes = [80.0] * 60  # Close price below lower band
        hist_df = self._make_hist_df(closes)

        rsi_series = pd.Series([50.0] * 60)
        macd_df = pd.DataFrame({
            "MACD_12_26_9": [1.0] * 60,
            "MACDh_12_26_9": [0.5] * 60,
            "MACDs_12_26_9": [0.5] * 60,
        })
        bbands_df = pd.DataFrame({
            "BBU_20_2.0": [110.0] * 60,
            "BBM_20_2.0": [100.0] * 60,
            "BBL_20_2.0": [90.0] * 60,  # Lower band at 90, close at 80 → BUY
        })

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist_df

        with patch("tools.technical_analyst.yfinance.Ticker", return_value=mock_ticker), \
             patch("tools.technical_analyst.ta.rsi", return_value=rsi_series), \
             patch("tools.technical_analyst.ta.macd", return_value=macd_df), \
             patch("tools.technical_analyst.ta.bbands", return_value=bbands_df):
            result = calculate_indicators("AAPL")

        assert "Bollinger: Price $80.00 / Upper $110.00 / Lower $90.00 → BUY (Below Lower Band)" in result

    def test_bollinger_sell_signal_above_upper(self):
        """calculate_indicators returns SELL when close > upper Bollinger Band."""
        closes = [120.0] * 60  # Close price above upper band
        hist_df = self._make_hist_df(closes)

        rsi_series = pd.Series([50.0] * 60)
        macd_df = pd.DataFrame({
            "MACD_12_26_9": [1.0] * 60,
            "MACDh_12_26_9": [0.5] * 60,
            "MACDs_12_26_9": [0.5] * 60,
        })
        bbands_df = pd.DataFrame({
            "BBU_20_2.0": [110.0] * 60,  # Upper band at 110, close at 120 → SELL
            "BBM_20_2.0": [100.0] * 60,
            "BBL_20_2.0": [90.0] * 60,
        })

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist_df

        with patch("tools.technical_analyst.yfinance.Ticker", return_value=mock_ticker), \
             patch("tools.technical_analyst.ta.rsi", return_value=rsi_series), \
             patch("tools.technical_analyst.ta.macd", return_value=macd_df), \
             patch("tools.technical_analyst.ta.bbands", return_value=bbands_df):
            result = calculate_indicators("AAPL")

        assert "Bollinger: Price $120.00 / Upper $110.00 / Lower $90.00 → SELL (Above Upper Band)" in result

    def test_invalid_ticker_returns_error(self):
        """calculate_indicators returns error for empty ticker."""
        result = calculate_indicators("")
        assert "Error" in result
        assert "Invalid ticker" in result

    def test_whitespace_ticker_returns_error(self):
        """calculate_indicators returns error for whitespace-only ticker."""
        result = calculate_indicators("   ")
        assert "Error" in result

    def test_ticker_not_found_returns_error(self):
        """calculate_indicators returns error when ticker not found (empty history)."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()

        with patch("tools.technical_analyst.yfinance.Ticker", return_value=mock_ticker):
            result = calculate_indicators("XYZZY")

        assert "Error" in result
        assert "not found" in result

    def test_exception_returns_error_string(self):
        """calculate_indicators catches exceptions and returns error string."""
        with patch("tools.technical_analyst.yfinance.Ticker", side_effect=Exception("Network error")):
            result = calculate_indicators("AAPL")

        assert "Error" in result
        assert "unexpected error" in result.lower()

    def test_cache_hit_returns_cached_result(self):
        """calculate_indicators returns cached result on second call."""
        closes = [100.0] * 60
        hist_df = self._make_hist_df(closes)

        rsi_series = pd.Series([50.0] * 60)
        macd_df = pd.DataFrame({
            "MACD_12_26_9": [1.0] * 60,
            "MACDh_12_26_9": [0.5] * 60,
            "MACDs_12_26_9": [0.5] * 60,
        })
        bbands_df = pd.DataFrame({
            "BBU_20_2.0": [110.0] * 60,
            "BBM_20_2.0": [100.0] * 60,
            "BBL_20_2.0": [90.0] * 60,
        })

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist_df

        with patch("tools.technical_analyst.yfinance.Ticker", return_value=mock_ticker) as mock_yf, \
             patch("tools.technical_analyst.ta.rsi", return_value=rsi_series), \
             patch("tools.technical_analyst.ta.macd", return_value=macd_df), \
             patch("tools.technical_analyst.ta.bbands", return_value=bbands_df):
            result1 = calculate_indicators("AAPL")
            result2 = calculate_indicators("AAPL")

        assert result1 == result2
        # yfinance.Ticker should only be called once due to caching
        assert mock_yf.call_count == 1

    def test_ticker_normalized_to_uppercase(self):
        """calculate_indicators normalizes ticker to uppercase."""
        closes = [100.0] * 60
        hist_df = self._make_hist_df(closes)

        rsi_series = pd.Series([50.0] * 60)
        macd_df = pd.DataFrame({
            "MACD_12_26_9": [1.0] * 60,
            "MACDh_12_26_9": [0.5] * 60,
            "MACDs_12_26_9": [0.5] * 60,
        })
        bbands_df = pd.DataFrame({
            "BBU_20_2.0": [110.0] * 60,
            "BBM_20_2.0": [100.0] * 60,
            "BBL_20_2.0": [90.0] * 60,
        })

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist_df

        with patch("tools.technical_analyst.yfinance.Ticker", return_value=mock_ticker) as mock_yf, \
             patch("tools.technical_analyst.ta.rsi", return_value=rsi_series), \
             patch("tools.technical_analyst.ta.macd", return_value=macd_df), \
             patch("tools.technical_analyst.ta.bbands", return_value=bbands_df):
            result = calculate_indicators("aapl")

        assert "Technical Signals for AAPL:" in result
        mock_yf.assert_called_once_with("AAPL")

    def test_output_format_matches_spec(self):
        """calculate_indicators output matches the specified format."""
        closes = [185.42] * 60
        hist_df = self._make_hist_df(closes)

        rsi_series = pd.Series([72.5] * 60)
        macd_values = [0.5] * 58 + [0.8, 1.24]
        signal_values = [0.5] * 58 + [0.99, 0.98]
        macd_df = pd.DataFrame({
            "MACD_12_26_9": macd_values,
            "MACDh_12_26_9": [0.0] * 60,
            "MACDs_12_26_9": signal_values,
        })
        bbands_df = pd.DataFrame({
            "BBU_20_2.0": [188.10] * 60,
            "BBM_20_2.0": [183.50] * 60,
            "BBL_20_2.0": [178.90] * 60,
        })

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist_df

        with patch("tools.technical_analyst.yfinance.Ticker", return_value=mock_ticker), \
             patch("tools.technical_analyst.ta.rsi", return_value=rsi_series), \
             patch("tools.technical_analyst.ta.macd", return_value=macd_df), \
             patch("tools.technical_analyst.ta.bbands", return_value=bbands_df):
            result = calculate_indicators("AAPL")

        # Verify the output format matches the spec example
        assert "Technical Signals for AAPL:" in result
        assert "RSI (14): 72.5 → SELL (Overbought)" in result
        assert "MACD: 1.24 / Signal: 0.98 → BUY (Bullish Crossover)" in result
        assert "Bollinger: Price $185.42 / Upper $188.10 / Lower $178.90 → NEUTRAL" in result



class TestGetPriceHistoryUnit:
    """Unit tests for get_price_history function.

    Validates: Requirements 11.4, 12.5
    """

    def setup_method(self):
        """Clear cache before each test."""
        cache.clear()

    def test_insufficient_data_shows_na_for_sma50(self):
        """get_price_history with only 30 days of data shows 'N/A' for SMA50.

        When fewer than 50 trading days are available, SMA50 cannot be calculated
        and should be marked as 'N/A', while other indicators with sufficient data
        (RSI needs 14, SMA20 needs 20) should still show values.

        Validates: Requirements 11.4
        """
        # Create 30 days of price data — enough for RSI(14) and SMA20, but not SMA50
        dates = pd.date_range(end="2024-01-15", periods=30, freq="B")
        closes = [100.0 + i * 0.5 for i in range(30)]
        hist_df = pd.DataFrame(
            {
                "Open": [c - 0.5 for c in closes],
                "High": [c + 1.0 for c in closes],
                "Low": [c - 1.0 for c in closes],
                "Close": closes,
                "Volume": [1000000] * 30,
            },
            index=dates,
        )

        # Create mock RSI series (14-period needs at least 14 data points — we have 30)
        rsi_series = pd.Series([55.0] * 30, index=dates)

        # Create mock MACD (needs 26 periods — we have 30, so some NaN at start)
        macd_df = pd.DataFrame(
            {
                "MACD_12_26_9": [float("nan")] * 25 + [0.5, 0.6, 0.7, 0.8, 0.9],
                "MACDh_12_26_9": [float("nan")] * 25 + [0.1, 0.1, 0.1, 0.1, 0.1],
                "MACDs_12_26_9": [float("nan")] * 25 + [0.4, 0.5, 0.6, 0.7, 0.8],
            },
            index=dates,
        )

        # Create mock SMA20 (needs 20 periods — we have 30)
        sma20_series = pd.Series(
            [float("nan")] * 19 + [100.0 + i * 0.5 for i in range(11)],
            index=dates,
        )

        # Create mock Bollinger Bands (needs 20 periods)
        bbands_df = pd.DataFrame(
            {
                "BBU_20_2.0": [float("nan")] * 19 + [115.0 + i * 0.5 for i in range(11)],
                "BBM_20_2.0": [float("nan")] * 19 + [105.0 + i * 0.5 for i in range(11)],
                "BBL_20_2.0": [float("nan")] * 19 + [95.0 + i * 0.5 for i in range(11)],
            },
            index=dates,
        )

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist_df

        with patch("tools.technical_analyst.yfinance.Ticker", return_value=mock_ticker), \
             patch("tools.technical_analyst.ta.rsi", return_value=rsi_series), \
             patch("tools.technical_analyst.ta.macd", return_value=macd_df), \
             patch("tools.technical_analyst.ta.sma", side_effect=[sma20_series, None]) as mock_sma, \
             patch("tools.technical_analyst.ta.bbands", return_value=bbands_df):
            result = get_price_history("AAPL")

        # Verify the output contains the header
        assert "Price History & Indicators for AAPL" in result

        # SMA50 should show "N/A" because we only have 30 days (< 50 required)
        # Check that N/A appears in the SMA50 column position for each row
        lines = result.strip().split("\n")
        data_lines = [l for l in lines if l and l[0].isdigit()]  # Lines starting with date
        for line in data_lines:
            # The SMA50 column should contain N/A
            parts = line.split("|")
            assert len(parts) >= 7, f"Expected at least 7 columns, got {len(parts)}: {line}"
            sma50_col = parts[6].strip()  # SMA50 is the 7th column (0-indexed: 6)
            assert sma50_col == "N/A", f"Expected SMA50 to be 'N/A', got '{sma50_col}'"

        # RSI should have values (not N/A) since we have 30 days > 14 required
        # Check at least one data line has a numeric RSI value
        has_rsi_value = False
        for line in data_lines:
            parts = line.split("|")
            rsi_col = parts[2].strip()
            if rsi_col != "N/A":
                has_rsi_value = True
                break
        assert has_rsi_value, "RSI should have values with 30 days of data"

        # SMA20 should have values since we have 30 days > 20 required
        has_sma20_value = False
        for line in data_lines:
            parts = line.split("|")
            sma20_col = parts[5].strip()
            if sma20_col != "N/A":
                has_sma20_value = True
                break
        assert has_sma20_value, "SMA20 should have values with 30 days of data"

    def test_insufficient_data_still_returns_formatted_output(self):
        """get_price_history with 30 days returns properly formatted table.

        Validates: Requirements 11.4
        """
        dates = pd.date_range(end="2024-01-15", periods=30, freq="B")
        closes = [150.0 + i * 0.2 for i in range(30)]
        hist_df = pd.DataFrame(
            {
                "Open": [c - 0.3 for c in closes],
                "High": [c + 0.8 for c in closes],
                "Low": [c - 0.8 for c in closes],
                "Close": closes,
                "Volume": [500000] * 30,
            },
            index=dates,
        )

        # RSI available (14 periods needed, 30 available)
        rsi_series = pd.Series([48.0] * 30, index=dates)

        # MACD available (26 periods needed, 30 available)
        macd_df = pd.DataFrame(
            {
                "MACD_12_26_9": [float("nan")] * 25 + [0.3, 0.4, 0.5, 0.6, 0.7],
                "MACDh_12_26_9": [float("nan")] * 25 + [0.1] * 5,
                "MACDs_12_26_9": [float("nan")] * 25 + [0.2, 0.3, 0.4, 0.5, 0.6],
            },
            index=dates,
        )

        # SMA20 available
        sma20_series = pd.Series(
            [float("nan")] * 19 + [150.0 + i * 0.2 for i in range(11)],
            index=dates,
        )

        # Bollinger Bands available
        bbands_df = pd.DataFrame(
            {
                "BBU_20_2.0": [float("nan")] * 19 + [160.0] * 11,
                "BBM_20_2.0": [float("nan")] * 19 + [155.0] * 11,
                "BBL_20_2.0": [float("nan")] * 19 + [150.0] * 11,
            },
            index=dates,
        )

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist_df

        with patch("tools.technical_analyst.yfinance.Ticker", return_value=mock_ticker), \
             patch("tools.technical_analyst.ta.rsi", return_value=rsi_series), \
             patch("tools.technical_analyst.ta.macd", return_value=macd_df), \
             patch("tools.technical_analyst.ta.sma", side_effect=[sma20_series, None]), \
             patch("tools.technical_analyst.ta.bbands", return_value=bbands_df):
            result = get_price_history("TSLA")

        # Should not contain "Error"
        assert "Error" not in result
        # Should have the header
        assert "Price History & Indicators for TSLA" in result
        assert "Date" in result
        assert "Close" in result
        assert "SMA50" in result


class TestCalculateIndicatorsAllNeutral:
    """Focused test for calculate_indicators with all-neutral scenario.

    Validates: Requirements 12.5
    """

    def setup_method(self):
        """Clear cache before each test."""
        cache.clear()

    def test_all_three_signals_neutral(self):
        """calculate_indicators returns NEUTRAL for RSI, MACD, and Bollinger when:
        - RSI = 50 (between 30 and 70)
        - MACD has no crossover (both previous and current MACD > signal)
        - Price is between Bollinger bands

        Validates: Requirements 12.5
        """
        closes = [100.0] * 60
        hist_df = pd.DataFrame({"Close": closes})

        # RSI = 50 → NEUTRAL (between 30 and 70)
        rsi_series = pd.Series([50.0] * 60)

        # MACD: no crossover — both previous and current MACD above signal
        # prev MACD (1.5) > prev signal (1.0) AND curr MACD (1.5) > curr signal (1.0) → NEUTRAL
        macd_df = pd.DataFrame({
            "MACD_12_26_9": [1.5] * 60,
            "MACDh_12_26_9": [0.5] * 60,
            "MACDs_12_26_9": [1.0] * 60,
        })

        # Bollinger: price (100) between lower (90) and upper (110) → NEUTRAL
        bbands_df = pd.DataFrame({
            "BBU_20_2.0": [110.0] * 60,
            "BBM_20_2.0": [100.0] * 60,
            "BBL_20_2.0": [90.0] * 60,
        })

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist_df

        with patch("tools.technical_analyst.yfinance.Ticker", return_value=mock_ticker), \
             patch("tools.technical_analyst.ta.rsi", return_value=rsi_series), \
             patch("tools.technical_analyst.ta.macd", return_value=macd_df), \
             patch("tools.technical_analyst.ta.bbands", return_value=bbands_df):
            result = calculate_indicators("AAPL")

        # Verify all three signals are NEUTRAL
        assert "RSI (14): 50.0 → NEUTRAL (Neutral)" in result
        assert "MACD: 1.50 / Signal: 1.00 → NEUTRAL (Neutral)" in result
        assert "Bollinger: Price $100.00 / Upper $110.00 / Lower $90.00 → NEUTRAL (Neutral)" in result

        # Count occurrences of NEUTRAL — should be exactly 3
        neutral_count = result.count("NEUTRAL")
        assert neutral_count == 3, f"Expected 3 NEUTRAL signals, got {neutral_count}"

        # Verify no BUY or SELL signals
        assert "BUY" not in result
        assert "SELL" not in result
