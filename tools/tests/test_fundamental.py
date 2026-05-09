"""
Tests for the Fundamental Analyst tools.

Covers:
- Graceful N/A for missing financial metrics (Property 11)
- Earnings surprise percentage formula (Property 12)
- Unit tests for specific scenarios
"""

import math
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from tools.fundamental_analyst import get_financials, get_earnings, cache


# ---------------------------------------------------------------------------
# Property 11: Graceful N/A for missing financial metrics
# Feature: agent-tools, Property 11: Graceful N/A for missing financial metrics
# Validates: Requirements 8.3
# ---------------------------------------------------------------------------

# The 5 financial metrics that may be None
FINANCIAL_METRICS = ["marketCap", "trailingPE", "revenueGrowth", "profitMargins", "debtToEquity"]

# Strategy: generate a random subset of metrics to set as None (at least 1)
none_subset_strategy = st.lists(
    st.sampled_from(FINANCIAL_METRICS),
    min_size=1,
    max_size=5,
    unique=True,
)


class TestGracefulNAForMissingMetrics:
    """Property 11: Graceful N/A for missing financial metrics.

    **Validates: Requirements 8.3**

    For any subset of the five financial metrics being None in the API response,
    get_financials SHALL return the available metrics with their values AND mark
    each unavailable metric as "N/A", without returning an error string.
    """

    def setup_method(self):
        """Clear cache before each test."""
        cache.clear()

    @settings(max_examples=100)
    @given(none_metrics=none_subset_strategy)
    @patch("tools.fundamental_analyst.yfinance.Ticker")
    def test_missing_metrics_show_na_no_error(self, mock_ticker_class, none_metrics):
        """
        Property 11: Graceful N/A for missing financial metrics.

        Generate random subsets of 5 metrics as None; verify "N/A" appears for
        missing metrics and no "Error" in output.

        **Validates: Requirements 8.3**
        """
        # Clear cache for each hypothesis example
        cache.clear()

        # Build a mock info dict with some metrics set to None
        info_dict = {
            "regularMarketPrice": 150.0,  # Ensures ticker is considered "found"
            "currentPrice": 150.0,
            "marketCap": 2_500_000_000_000,
            "trailingPE": 28.5,
            "revenueGrowth": 0.082,
            "profitMargins": 0.253,
            "debtToEquity": 1.73,
        }

        # Set the randomly chosen metrics to None
        for metric in none_metrics:
            info_dict[metric] = None

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = info_dict
        mock_ticker_class.return_value = mock_ticker_instance

        result = get_financials("AAPL")

        # Verify: output does NOT contain "Error" (graceful degradation)
        assert "Error" not in result, (
            f"Output should not contain 'Error' when metrics are missing. "
            f"Got: {result}"
        )

        # Verify: for each metric set to None, "N/A" appears in the output
        assert "N/A" in result, (
            f"Output should contain 'N/A' for missing metrics {none_metrics}. "
            f"Got: {result}"
        )

        # Count the number of "N/A" occurrences - should be at least len(none_metrics)
        na_count = result.count("N/A")
        assert na_count >= len(none_metrics), (
            f"Expected at least {len(none_metrics)} 'N/A' occurrences for "
            f"missing metrics {none_metrics}, but found {na_count}. Got: {result}"
        )


class TestGetEarningsUnit:
    """Unit tests for get_earnings function."""

    def setup_method(self):
        """Clear cache before each test."""
        cache.clear()

    def test_crypto_ticker_returns_not_available(self):
        """Crypto tickers (containing '-') return 'not available' message."""
        result = get_earnings("BTC-USD")
        assert "not available for this instrument type" in result

    def test_crypto_ticker_eth_returns_not_available(self):
        """ETH-USD also returns 'not available' message."""
        result = get_earnings("ETH-USD")
        assert "not available for this instrument type" in result

    def test_invalid_empty_ticker_returns_error(self):
        """Empty ticker returns error message."""
        result = get_earnings("")
        assert "Error" in result

    def test_invalid_whitespace_ticker_returns_error(self):
        """Whitespace-only ticker returns error message."""
        result = get_earnings("   ")
        assert "Error" in result

    @patch("tools.fundamental_analyst.yfinance.Ticker")
    def test_valid_ticker_with_earnings_data(self, mock_ticker_class):
        """Valid ticker with earnings data returns formatted output."""
        # Create mock earnings_dates DataFrame
        dates = pd.to_datetime([
            "2024-04-25", "2024-01-25", "2023-10-26", "2023-07-27"
        ])
        earnings_df = pd.DataFrame(
            {
                "EPS Estimate": [1.50, 2.10, 1.39, 1.19],
                "Reported EPS": [1.53, 2.18, 1.46, 1.26],
                "Surprise(%)": [2.0, 3.81, 5.04, 5.88],
            },
            index=dates,
        )

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.earnings_dates = earnings_df
        mock_ticker_class.return_value = mock_ticker_instance

        result = get_earnings("AAPL")

        assert "Earnings History for AAPL" in result
        assert "Last 4 Quarters" in result
        # Check surprise calculation for first quarter
        # ((1.53 - 1.50) / |1.50|) * 100 = 2.00%
        assert "+2.00%" in result

    @patch("tools.fundamental_analyst.yfinance.Ticker")
    def test_surprise_calculation_positive(self, mock_ticker_class):
        """Positive surprise is calculated correctly."""
        dates = pd.to_datetime(["2024-04-25"])
        earnings_df = pd.DataFrame(
            {
                "EPS Estimate": [1.50],
                "Reported EPS": [1.53],
            },
            index=dates,
        )

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.earnings_dates = earnings_df
        mock_ticker_class.return_value = mock_ticker_instance

        result = get_earnings("AAPL")

        # ((1.53 - 1.50) / |1.50|) * 100 = 2.00%
        assert "+2.00%" in result

    @patch("tools.fundamental_analyst.yfinance.Ticker")
    def test_surprise_calculation_negative(self, mock_ticker_class):
        """Negative surprise is calculated correctly."""
        dates = pd.to_datetime(["2024-04-25"])
        earnings_df = pd.DataFrame(
            {
                "EPS Estimate": [2.00],
                "Reported EPS": [1.80],
            },
            index=dates,
        )

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.earnings_dates = earnings_df
        mock_ticker_class.return_value = mock_ticker_instance

        result = get_earnings("AAPL")

        # ((1.80 - 2.00) / |2.00|) * 100 = -10.00%
        assert "-10.00%" in result

    @patch("tools.fundamental_analyst.yfinance.Ticker")
    def test_nan_estimate_shows_na(self, mock_ticker_class):
        """NaN estimated EPS shows N/A for surprise."""
        dates = pd.to_datetime(["2024-04-25"])
        earnings_df = pd.DataFrame(
            {
                "EPS Estimate": [float("nan")],
                "Reported EPS": [1.53],
            },
            index=dates,
        )

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.earnings_dates = earnings_df
        mock_ticker_class.return_value = mock_ticker_instance

        result = get_earnings("AAPL")

        assert "N/A" in result

    @patch("tools.fundamental_analyst.yfinance.Ticker")
    def test_zero_estimate_shows_na(self, mock_ticker_class):
        """Zero estimated EPS shows N/A for surprise (avoid division by zero)."""
        dates = pd.to_datetime(["2024-04-25"])
        earnings_df = pd.DataFrame(
            {
                "EPS Estimate": [0.0],
                "Reported EPS": [1.53],
            },
            index=dates,
        )

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.earnings_dates = earnings_df
        mock_ticker_class.return_value = mock_ticker_instance

        result = get_earnings("AAPL")

        assert "N/A" in result

    @patch("tools.fundamental_analyst.yfinance.Ticker")
    def test_empty_earnings_returns_not_available(self, mock_ticker_class):
        """Empty earnings DataFrame returns 'not available' message."""
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.earnings_dates = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker_instance

        result = get_earnings("AAPL")

        assert "not available for this instrument type" in result

    @patch("tools.fundamental_analyst.yfinance.Ticker")
    def test_none_earnings_returns_not_available(self, mock_ticker_class):
        """None earnings_dates returns 'not available' message."""
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.earnings_dates = None
        mock_ticker_class.return_value = mock_ticker_instance

        result = get_earnings("AAPL")

        assert "not available for this instrument type" in result

    @patch("tools.fundamental_analyst.yfinance.Ticker")
    def test_only_4_quarters_returned(self, mock_ticker_class):
        """Only the most recent 4 quarters are returned even if more data exists."""
        dates = pd.to_datetime([
            "2024-07-25", "2024-04-25", "2024-01-25",
            "2023-10-26", "2023-07-27", "2023-04-27"
        ])
        earnings_df = pd.DataFrame(
            {
                "EPS Estimate": [1.50, 1.40, 2.10, 1.39, 1.19, 1.10],
                "Reported EPS": [1.53, 1.45, 2.18, 1.46, 1.26, 1.15],
            },
            index=dates,
        )

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.earnings_dates = earnings_df
        mock_ticker_class.return_value = mock_ticker_instance

        result = get_earnings("AAPL")

        # Count the number of quarter lines (lines with "EPS $")
        eps_lines = [line for line in result.split("\n") if "EPS $" in line]
        assert len(eps_lines) == 4

    @patch("tools.fundamental_analyst.yfinance.Ticker")
    def test_cache_hit_returns_cached_result(self, mock_ticker_class):
        """Second call with same ticker returns cached result without API call."""
        dates = pd.to_datetime(["2024-04-25"])
        earnings_df = pd.DataFrame(
            {
                "EPS Estimate": [1.50],
                "Reported EPS": [1.53],
            },
            index=dates,
        )

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.earnings_dates = earnings_df
        mock_ticker_class.return_value = mock_ticker_instance

        # First call
        result1 = get_earnings("AAPL")
        # Second call should use cache
        result2 = get_earnings("AAPL")

        assert result1 == result2
        # Ticker should only be instantiated once (second call uses cache)
        assert mock_ticker_class.call_count == 1

    @patch("tools.fundamental_analyst.yfinance.Ticker")
    def test_exception_returns_error_string(self, mock_ticker_class):
        """Unexpected exceptions are caught and return error string."""
        mock_ticker_class.side_effect = RuntimeError("Network failure")

        result = get_earnings("AAPL")

        assert "Error" in result
        assert "Network failure" in result

    def test_ticker_normalized_to_uppercase(self):
        """Ticker is normalized to uppercase before processing."""
        # Crypto detection should work with lowercase too
        result = get_earnings("btc-usd")
        assert "not available for this instrument type" in result

    @patch("tools.fundamental_analyst.yfinance.Ticker")
    def test_quarter_label_format(self, mock_ticker_class):
        """Quarter labels are formatted as Q1-Q4 with year."""
        dates = pd.to_datetime(["2024-04-25", "2024-01-25"])
        earnings_df = pd.DataFrame(
            {
                "EPS Estimate": [1.50, 2.10],
                "Reported EPS": [1.53, 2.18],
            },
            index=dates,
        )

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.earnings_dates = earnings_df
        mock_ticker_class.return_value = mock_ticker_instance

        result = get_earnings("AAPL")

        # April = Q2, January = Q1
        assert "Q2 2024" in result
        assert "Q1 2024" in result


# --- Property-Based Tests ---

from hypothesis import given, settings, assume
from hypothesis import strategies as st


class TestEarningsSurpriseProperty:
    """
    Property 12: Earnings surprise percentage formula correctness

    For any pair (reported_EPS, estimated_EPS) where estimated_EPS ≠ 0,
    the surprise percentage SHALL equal
    round(((reported_EPS - estimated_EPS) / abs(estimated_EPS)) * 100, 2).

    Feature: agent-tools, Property 12: Earnings surprise percentage formula correctness
    **Validates: Requirements 9.3**
    """

    def setup_method(self):
        """Clear cache before each test."""
        cache.clear()

    @given(
        reported_eps=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        estimated_eps=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    @patch("tools.fundamental_analyst.yfinance.Ticker")
    def test_surprise_percentage_formula(self, mock_ticker_class, reported_eps, estimated_eps):
        """
        Property 12: Earnings surprise percentage formula correctness.

        Generate random (reported_EPS, estimated_EPS) pairs where estimated ≠ 0;
        verify surprise formula matches round(((reported - estimated) / abs(estimated)) * 100, 2).

        Feature: agent-tools, Property 12: Earnings surprise percentage formula correctness
        **Validates: Requirements 9.3**
        """
        # Ensure estimated_eps is not zero (avoid division by zero)
        assume(estimated_eps != 0.0)
        assume(abs(estimated_eps) > 1e-9)

        # Clear cache to avoid stale results between hypothesis examples
        cache.clear()

        # Create mock earnings_dates DataFrame with the generated EPS values
        dates = pd.to_datetime(["2024-04-25"])
        earnings_df = pd.DataFrame(
            {
                "EPS Estimate": [estimated_eps],
                "Reported EPS": [reported_eps],
            },
            index=dates,
        )

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.earnings_dates = earnings_df
        mock_ticker_class.return_value = mock_ticker_instance

        result = get_earnings("AAPL")

        # Calculate expected surprise percentage
        expected_surprise = round(((reported_eps - estimated_eps) / abs(estimated_eps)) * 100, 2)

        # The result should contain the surprise percentage formatted correctly
        if expected_surprise >= 0:
            expected_str = f"+{expected_surprise:.2f}%"
        else:
            expected_str = f"{expected_surprise:.2f}%"

        assert expected_str in result, (
            f"Expected surprise '{expected_str}' not found in result.\n"
            f"reported_eps={reported_eps}, estimated_eps={estimated_eps}\n"
            f"Result: {result}"
        )


# --- Task 6.6: Unit tests for fundamental analyst tools ---
from tools.fundamental_analyst import get_financials, get_peers


class TestGetFinancialsUnit:
    """Unit tests for get_financials function.

    Validates: Requirements 8.2
    """

    def setup_method(self):
        """Clear cache before each test."""
        cache.clear()

    @patch("tools.fundamental_analyst.yfinance.Ticker")
    def test_all_fields_present(self, mock_ticker_class):
        """When all 5 financial metrics are present, output contains each with correct formatting."""
        mock_info = {
            "regularMarketPrice": 185.42,
            "currentPrice": 185.42,
            "marketCap": 2_850_000_000_000,  # $2.85T
            "trailingPE": 28.5,
            "revenueGrowth": 0.082,  # 8.20%
            "profitMargins": 0.253,  # 25.30%
            "debtToEquity": 1.73,
        }

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = mock_info
        mock_ticker_class.return_value = mock_ticker_instance

        result = get_financials("AAPL")

        # Verify header
        assert "Financial Metrics for AAPL" in result

        # Verify market cap formatted with B unit
        assert "Market Cap:" in result
        assert "$2850.00B" in result or "$2.85" in result

        # Verify P/E ratio
        assert "P/E Ratio:" in result
        assert "28.5" in result

        # Verify revenue growth as percentage
        assert "Revenue Growth:" in result
        assert "8.20%" in result

        # Verify profit margin as percentage
        assert "Profit Margin:" in result
        assert "25.30%" in result

        # Verify debt/equity
        assert "Debt/Equity:" in result
        assert "1.73" in result

        # Verify no error in output
        assert "Error" not in result


class TestGetEarningsCryptoUnit:
    """Unit tests for get_earnings with crypto tickers.

    Validates: Requirements 9.4
    """

    def setup_method(self):
        """Clear cache before each test."""
        cache.clear()

    def test_btc_usd_returns_not_available(self):
        """BTC-USD returns 'not available for this instrument type' message."""
        result = get_earnings("BTC-USD")
        assert "not available for this instrument type" in result

    def test_eth_usd_returns_not_available(self):
        """ETH-USD returns 'not available for this instrument type' message."""
        result = get_earnings("ETH-USD")
        assert "not available for this instrument type" in result

    def test_lowercase_crypto_returns_not_available(self):
        """Lowercase crypto ticker also returns 'not available' (normalization works)."""
        result = get_earnings("btc-usd")
        assert "not available for this instrument type" in result


class TestGetPeersUnit:
    """Unit tests for get_peers function.

    Validates: Requirements 10.1, 10.2, 10.3
    """

    def setup_method(self):
        """Clear cache before each test."""
        cache.clear()

    def test_btc_usd_returns_not_available(self):
        """BTC-USD returns 'not available for this instrument type' message."""
        result = get_peers("BTC-USD")
        assert "not available for this instrument type" in result

    def test_eth_usd_returns_not_available(self):
        """ETH-USD returns 'not available for this instrument type' message."""
        result = get_peers("ETH-USD")
        assert "not available for this instrument type" in result

    def test_lowercase_crypto_returns_not_available(self):
        """Lowercase crypto ticker also returns 'not available' (normalization works)."""
        result = get_peers("btc-usd")
        assert "not available for this instrument type" in result

    @patch("tools.fundamental_analyst.yfinance.Ticker")
    def test_valid_stock_returns_sector_industry_peers(self, mock_ticker_class):
        """Valid stock ticker returns sector, industry, and up to 5 peers."""
        mock_info = {
            "regularMarketPrice": 185.0,
            "currentPrice": 185.0,
            "sector": "Technology",
            "industry": "Consumer Electronics",
        }
        mock_instance = MagicMock()
        mock_instance.info = mock_info
        mock_ticker_class.return_value = mock_instance

        result = get_peers("AAPL")

        assert "Peer Analysis for AAPL" in result
        assert "Sector: Technology" in result
        assert "Industry: Consumer Electronics" in result
        assert "Peers:" in result
        # Should have up to 5 peers, excluding AAPL itself
        peer_lines = [line for line in result.split("\n") if line.startswith("- ")]
        assert len(peer_lines) <= 5
        assert len(peer_lines) > 0
        # AAPL should not be in its own peer list
        assert "- AAPL" not in result

    @patch("tools.fundamental_analyst.yfinance.Ticker")
    def test_etf_no_sector_returns_not_available(self, mock_ticker_class):
        """ETF with no sector field returns 'not available' message."""
        mock_info = {
            "regularMarketPrice": 500.0,
            "currentPrice": 500.0,
            # No 'sector' key
        }
        mock_instance = MagicMock()
        mock_instance.info = mock_info
        mock_ticker_class.return_value = mock_instance

        result = get_peers("SPY")

        assert "not available for this instrument type" in result

    @patch("tools.fundamental_analyst.yfinance.Ticker")
    def test_peers_limited_to_5(self, mock_ticker_class):
        """Peer list is limited to at most 5 companies."""
        mock_info = {
            "regularMarketPrice": 150.0,
            "currentPrice": 150.0,
            "sector": "Technology",
            "industry": "Software - Infrastructure",
        }
        mock_instance = MagicMock()
        mock_instance.info = mock_info
        mock_ticker_class.return_value = mock_instance

        # Use a ticker NOT in the Technology list so all 10 are candidates
        result = get_peers("PLTR")

        peer_lines = [line for line in result.split("\n") if line.startswith("- ")]
        assert len(peer_lines) <= 5

    @patch("tools.fundamental_analyst.yfinance.Ticker")
    def test_cache_hit_returns_cached_result(self, mock_ticker_class):
        """Second call with same ticker returns cached result without API call."""
        mock_info = {
            "regularMarketPrice": 185.0,
            "currentPrice": 185.0,
            "sector": "Technology",
            "industry": "Consumer Electronics",
        }
        mock_instance = MagicMock()
        mock_instance.info = mock_info
        mock_ticker_class.return_value = mock_instance

        result1 = get_peers("AAPL")
        result2 = get_peers("AAPL")

        assert result1 == result2
        # Ticker should only be instantiated once (second call uses cache)
        assert mock_ticker_class.call_count == 1

    @patch("tools.fundamental_analyst.yfinance.Ticker")
    def test_exception_returns_error_string(self, mock_ticker_class):
        """Unexpected exceptions are caught and return error string."""
        mock_ticker_class.side_effect = RuntimeError("Connection timeout")

        result = get_peers("AAPL")

        assert "Error" in result
        assert "Connection timeout" in result

    def test_invalid_empty_ticker_returns_error(self):
        """Empty ticker returns error message."""
        result = get_peers("")
        assert "Error" in result
        assert "Invalid ticker" in result

    def test_invalid_whitespace_ticker_returns_error(self):
        """Whitespace-only ticker returns error message."""
        result = get_peers("   ")
        assert "Error" in result

    @patch("tools.fundamental_analyst.yfinance.Ticker")
    def test_unknown_sector_shows_no_peer_data(self, mock_ticker_class):
        """Ticker with a sector not in SECTOR_PEERS mapping shows no peer data message."""
        mock_info = {
            "regularMarketPrice": 50.0,
            "currentPrice": 50.0,
            "sector": "Exotic Sector",
            "industry": "Niche Industry",
        }
        mock_instance = MagicMock()
        mock_instance.info = mock_info
        mock_ticker_class.return_value = mock_instance

        result = get_peers("XYZ")

        assert "Sector: Exotic Sector" in result
        assert "Industry: Niche Industry" in result
        assert "No peer data available" in result
