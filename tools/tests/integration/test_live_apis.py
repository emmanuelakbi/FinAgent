"""
Integration tests against live APIs.

These tests require network access and may be rate-limited.
Run separately from CI with: pytest -m integration

Covers:
- Live API calls for all 10 tool functions
- End-to-end verification of data retrieval and formatting
"""

import time

import pytest

from tools.market_scanner import search_news, get_price_change, get_volume
from tools.fundamental_analyst import get_financials, get_earnings, get_peers
from tools.technical_analyst import get_price_history, calculate_indicators
from tools.risk_manager import set_stop_loss

# Check if pandas_ta is available (not mocked)
try:
    import pandas_ta

    _pandas_ta_available = not hasattr(pandas_ta, "_mock_name")
except ImportError:
    _pandas_ta_available = False

requires_pandas_ta = pytest.mark.skipif(
    not _pandas_ta_available,
    reason="pandas_ta not available (mocked or not installed)",
)

pytestmark = pytest.mark.integration


class TestSearchNewsLive:
    """Integration tests for search_news against DuckDuckGo."""

    def test_search_news_aapl_returns_non_error(self):
        """Validates: Requirements 5.1"""
        result = search_news("AAPL")
        assert isinstance(result, str)
        # DuckDuckGo may rate-limit; skip if rate limited
        if "Ratelimit" in result or "rate limit" in result.lower():
            pytest.skip("DuckDuckGo rate limited - transient failure")
        assert "Error" not in result
        assert "AAPL" in result


class TestGetPriceChangeLive:
    """Integration tests for get_price_change against yfinance."""

    def test_get_price_change_nvda_returns_price_data(self):
        """Validates: Requirements 4.1, 6.1"""
        time.sleep(1)
        result = get_price_change("NVDA")
        assert isinstance(result, str)
        assert "Error" not in result
        assert "$" in result
        assert "NVDA" in result

    def test_get_price_change_btc_usd_returns_price_data(self):
        """Validates: Requirements 4.2, 6.1"""
        time.sleep(1)
        result = get_price_change("BTC-USD")
        assert isinstance(result, str)
        assert "Error" not in result
        assert "$" in result
        assert "BTC-USD" in result


class TestGetVolumeLive:
    """Integration tests for get_volume against yfinance."""

    def test_get_volume_aapl_returns_volume_data(self):
        """Validates: Requirements 7.1"""
        time.sleep(1)
        result = get_volume("AAPL")
        assert isinstance(result, str)
        assert "Error" not in result
        assert "Volume" in result
        assert "AAPL" in result


class TestGetFinancialsLive:
    """Integration tests for get_financials against yfinance."""

    def test_get_financials_aapl_returns_all_metrics(self):
        """Validates: Requirements 8.1"""
        time.sleep(1)
        result = get_financials("AAPL")
        assert isinstance(result, str)
        assert "Error" not in result
        assert "AAPL" in result
        # Verify all 5 metrics are present
        assert "Market Cap" in result
        assert "P/E Ratio" in result
        assert "Revenue Growth" in result
        assert "Profit Margin" in result
        assert "Debt/Equity" in result


class TestGetEarningsLive:
    """Integration tests for get_earnings against yfinance."""

    def test_get_earnings_aapl_returns_quarters(self):
        """Validates: Requirements 9.1"""
        time.sleep(1)
        result = get_earnings("AAPL")
        assert isinstance(result, str)
        assert "Error" not in result
        assert "AAPL" in result
        assert "EPS" in result


class TestGetPeersLive:
    """Integration tests for get_peers against yfinance."""

    def test_get_peers_aapl_returns_sector_and_peers(self):
        """Validates: Requirements 10.1"""
        time.sleep(1)
        result = get_peers("AAPL")
        assert isinstance(result, str)
        assert "Error" not in result
        assert "AAPL" in result
        assert "Sector" in result or "sector" in result.lower()


class TestGetPriceHistoryLive:
    """Integration tests for get_price_history against yfinance."""

    @requires_pandas_ta
    def test_get_price_history_aapl_returns_indicators(self):
        """Validates: Requirements 11.1"""
        time.sleep(1)
        result = get_price_history("AAPL")
        assert isinstance(result, str)
        assert "Error" not in result
        assert "AAPL" in result
        # Should contain technical indicator labels
        assert "RSI" in result or "SMA" in result or "MACD" in result


class TestCalculateIndicatorsLive:
    """Integration tests for calculate_indicators against yfinance."""

    @requires_pandas_ta
    def test_calculate_indicators_aapl_returns_signals(self):
        """Validates: Requirements 12.1"""
        time.sleep(1)
        result = calculate_indicators("AAPL")
        assert isinstance(result, str)
        assert "Error" not in result
        assert "AAPL" in result
        # Should contain all 3 signal classifications
        assert "RSI" in result
        assert "MACD" in result
        assert "Bollinger" in result or "BB" in result


class TestSetStopLossLive:
    """Integration tests for set_stop_loss against yfinance."""

    @requires_pandas_ta
    def test_set_stop_loss_aapl_returns_levels(self):
        """Validates: Requirements 14.1"""
        time.sleep(1)
        result = set_stop_loss("AAPL", 185.0, 1.5)
        assert isinstance(result, str)
        assert "Error" not in result
        assert "AAPL" in result
        assert "Stop Loss" in result or "Stop-Loss" in result
        assert "Take Profit" in result or "Take-Profit" in result
        assert "ATR" in result
