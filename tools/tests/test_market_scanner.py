"""
Tests for the Market Scanner tools.

Covers:
- search_news format and bounds (Property 8)
- get_price_change percentage formula (Property 9)
- get_volume ratio and UNUSUAL VOLUME flag (Property 10)
- Unit tests for specific scenarios
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from tools.market_scanner import search_news, get_price_change, get_volume, cache


class TestSearchNewsUnit:
    """Unit tests for search_news function."""

    def setup_method(self):
        """Clear cache before each test."""
        cache.clear()

    def test_valid_ticker_with_results(self):
        """search_news returns formatted output with numbered articles."""
        mock_articles = [
            {"title": "Apple Reports Record Revenue", "body": "Apple Inc. reported record Q4 revenue."},
            {"title": "AAPL Stock Rises", "body": "Shares of Apple rose 3% today."},
            {"title": "New iPhone Launch", "body": "Apple announced the new iPhone model."},
        ]

        with patch("tools.market_scanner.DDGS") as mock_ddgs_class:
            mock_ddgs_instance = MagicMock()
            mock_ddgs_instance.news.return_value = mock_articles
            mock_ddgs_class.return_value = mock_ddgs_instance

            result = search_news("AAPL")

        assert "Recent News for AAPL (last 7 days):" in result
        assert "1. Apple Reports Record Revenue - Apple Inc. reported record Q4 revenue." in result
        assert "2. AAPL Stock Rises - Shares of Apple rose 3% today." in result
        assert "3. New iPhone Launch - Apple announced the new iPhone model." in result

    def test_empty_results_returns_no_news_message(self):
        """search_news returns 'no recent news' message when no articles found."""
        with patch("tools.market_scanner.DDGS") as mock_ddgs_class:
            mock_ddgs_instance = MagicMock()
            mock_ddgs_instance.news.return_value = []
            mock_ddgs_class.return_value = mock_ddgs_instance

            result = search_news("XYZZY")

        assert "No recent news found for XYZZY" in result
        assert "7 days" in result

    def test_max_5_articles(self):
        """search_news returns at most 5 articles even if more are available."""
        mock_articles = [
            {"title": f"Article {i}", "body": f"Body {i}"} for i in range(10)
        ]

        with patch("tools.market_scanner.DDGS") as mock_ddgs_class:
            mock_ddgs_instance = MagicMock()
            mock_ddgs_instance.news.return_value = mock_articles
            mock_ddgs_class.return_value = mock_ddgs_instance

            result = search_news("AAPL")

        assert "5." in result
        assert "6." not in result

    def test_invalid_ticker_returns_error(self):
        """search_news returns error for empty ticker."""
        result = search_news("")
        assert "Error" in result
        assert "Invalid ticker" in result

    def test_whitespace_ticker_returns_error(self):
        """search_news returns error for whitespace-only ticker."""
        result = search_news("   ")
        assert "Error" in result

    def test_ticker_normalized_to_uppercase(self):
        """search_news normalizes ticker to uppercase."""
        with patch("tools.market_scanner.DDGS") as mock_ddgs_class:
            mock_ddgs_instance = MagicMock()
            mock_ddgs_instance.news.return_value = [
                {"title": "Test", "body": "Test body"}
            ]
            mock_ddgs_class.return_value = mock_ddgs_instance

            result = search_news("aapl")

        assert "Recent News for AAPL" in result
        mock_ddgs_instance.news.assert_called_once_with(
            query="AAPL", max_results=5, timelimit="w"
        )

    def test_cache_hit_returns_cached_result(self):
        """search_news returns cached result on second call."""
        mock_articles = [
            {"title": "Test Article", "body": "Test body"}
        ]

        with patch("tools.market_scanner.DDGS") as mock_ddgs_class:
            mock_ddgs_instance = MagicMock()
            mock_ddgs_instance.news.return_value = mock_articles
            mock_ddgs_class.return_value = mock_ddgs_instance

            result1 = search_news("AAPL")
            result2 = search_news("AAPL")

        assert result1 == result2
        # DDGS should only be instantiated once due to caching
        assert mock_ddgs_class.call_count == 1

    def test_exception_returns_error_string(self):
        """search_news catches exceptions and returns error string."""
        with patch("tools.market_scanner.DDGS", side_effect=Exception("Network error")):
            result = search_news("AAPL")

        assert "Error" in result
        assert "unexpected error" in result.lower()

    def test_uses_7_day_timelimit(self):
        """search_news uses timelimit='w' for 7-day window."""
        with patch("tools.market_scanner.DDGS") as mock_ddgs_class:
            mock_ddgs_instance = MagicMock()
            mock_ddgs_instance.news.return_value = []
            mock_ddgs_class.return_value = mock_ddgs_instance

            search_news("AAPL")

        mock_ddgs_instance.news.assert_called_once_with(
            query="AAPL", max_results=5, timelimit="w"
        )


class TestGetPriceChangeUnit:
    """Unit tests for get_price_change function."""

    def setup_method(self):
        """Clear cache before each test."""
        cache.clear()

    def test_valid_ticker_returns_formatted_output(self):
        """get_price_change returns correctly formatted output for a valid ticker."""
        mock_info = {"currentPrice": 185.42, "previousClose": 183.10}
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info

        with patch("tools.market_scanner.yfinance.Ticker", return_value=mock_ticker):
            result = get_price_change("AAPL")

        assert "Price Change for AAPL:" in result
        assert "Current Price: $185.42" in result
        assert "Previous Close: $183.10" in result
        assert "Change: +$2.32 (+1.27%)" in result

    def test_negative_change_shows_minus_sign(self):
        """get_price_change shows minus sign for negative price changes."""
        mock_info = {"currentPrice": 180.00, "previousClose": 185.00}
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info

        with patch("tools.market_scanner.yfinance.Ticker", return_value=mock_ticker):
            result = get_price_change("AAPL")

        assert "Change: -$5.00 (-2.7%)" in result

    def test_fallback_to_history_when_info_missing(self):
        """get_price_change falls back to history(period='2d') when info fields are None."""
        mock_info = {"currentPrice": None, "previousClose": None}
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info

        # Create a DataFrame with 2 days of history
        hist_data = pd.DataFrame({"Close": [183.10, 185.42]})
        mock_ticker.history.return_value = hist_data

        with patch("tools.market_scanner.yfinance.Ticker", return_value=mock_ticker):
            result = get_price_change("AAPL")

        assert "Price Change for AAPL:" in result
        assert "Current Price: $185.42" in result
        assert "Previous Close: $183.10" in result
        mock_ticker.history.assert_called_once_with(period="2d")

    def test_invalid_ticker_returns_error(self):
        """get_price_change returns error for empty ticker."""
        result = get_price_change("")
        assert "Error" in result
        assert "Invalid ticker" in result

    def test_whitespace_ticker_returns_error(self):
        """get_price_change returns error for whitespace-only ticker."""
        result = get_price_change("   ")
        assert "Error" in result
        assert "Invalid ticker" in result

    def test_ticker_not_found_returns_error(self):
        """get_price_change returns error when ticker not found (empty history fallback)."""
        mock_info = {"currentPrice": None, "previousClose": None}
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info
        mock_ticker.history.return_value = pd.DataFrame()

        with patch("tools.market_scanner.yfinance.Ticker", return_value=mock_ticker):
            result = get_price_change("XYZZY")

        assert "Error" in result
        assert "not found" in result

    def test_exception_returns_error_string(self):
        """get_price_change catches exceptions and returns error string."""
        with patch("tools.market_scanner.yfinance.Ticker", side_effect=Exception("Network error")):
            result = get_price_change("AAPL")

        assert "Error" in result
        assert "unexpected error" in result.lower()

    def test_cache_hit_returns_cached_result(self):
        """get_price_change returns cached result on second call."""
        mock_info = {"currentPrice": 185.42, "previousClose": 183.10}
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info

        with patch("tools.market_scanner.yfinance.Ticker", return_value=mock_ticker) as mock_yf:
            result1 = get_price_change("AAPL")
            result2 = get_price_change("AAPL")

        assert result1 == result2
        # yfinance.Ticker should only be called once due to caching
        assert mock_yf.call_count == 1

    def test_ticker_normalized_to_uppercase(self):
        """get_price_change normalizes ticker to uppercase."""
        mock_info = {"currentPrice": 185.42, "previousClose": 183.10}
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info

        with patch("tools.market_scanner.yfinance.Ticker", return_value=mock_ticker) as mock_yf:
            result = get_price_change("aapl")

        assert "Price Change for AAPL:" in result
        mock_yf.assert_called_once_with("AAPL")

    def test_zero_change(self):
        """get_price_change handles zero change correctly."""
        mock_info = {"currentPrice": 100.00, "previousClose": 100.00}
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info

        with patch("tools.market_scanner.yfinance.Ticker", return_value=mock_ticker):
            result = get_price_change("AAPL")

        assert "Change: +$0.00 (+0.0%)" in result


class TestGetPriceChangeProperty:
    """Property-based tests for get_price_change percentage formula.

    Feature: agent-tools, Property 9: Price change percentage formula correctness

    **Validates: Requirements 6.3**

    For any pair of prices (current_price, previous_close) where previous_close > 0,
    the percentage change reported by get_price_change SHALL equal
    round(((current_price - previous_close) / previous_close) * 100, 2).
    """

    def setup_method(self):
        """Clear cache before each test."""
        cache.clear()

    @given(
        current_price=st.floats(min_value=0.01, max_value=100000.0, allow_nan=False, allow_infinity=False),
        previous_close=st.floats(min_value=0.01, max_value=100000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_percentage_formula_correctness(self, current_price, previous_close):
        """The percentage change matches the formula: ((current - previous) / previous) * 100 rounded to 2 decimals.

        **Validates: Requirements 6.3**
        """
        cache.clear()

        expected_percent = round(((current_price - previous_close) / previous_close) * 100, 2)

        mock_info = {"currentPrice": current_price, "previousClose": previous_close}
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info

        with patch("tools.market_scanner.yfinance.Ticker", return_value=mock_ticker):
            result = get_price_change("TEST")

        # The result should contain the expected percentage
        assert f"{'+' if expected_percent >= 0 else ''}{expected_percent}%" in result, (
            f"Expected percentage {expected_percent}% not found in result: {result}"
        )


class TestSearchNewsProperty:
    """Property-based tests for search_news results bounded and date-filtered.

    Feature: agent-tools, Property 8: News results bounded and date-filtered

    **Validates: Requirements 5.2, 5.3**

    For any list of N news articles returned by DuckDuckGo (where N >= 0),
    the formatted output of search_news SHALL contain at most 5 article entries,
    and every included article SHALL have a publication date within the most recent 7 days.
    """

    def setup_method(self):
        """Clear cache before each test."""
        cache.clear()

    @given(
        news_items=st.lists(
            st.fixed_dictionaries({
                "title": st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"))),
                "body": st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"))),
                "date": st.dates().map(lambda d: d.strftime("%Y-%m-%dT%H:%M:%S+00:00")),
                "url": st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"))),
            }),
            min_size=0,
            max_size=20,
        )
    )
    @settings(max_examples=100)
    def test_news_results_bounded_and_date_filtered(self, news_items):
        """Output has at most 5 numbered articles and the API enforces 7-day filtering.

        Generates lists of 0-20 news items with random titles, bodies, and dates.
        Mocks the DDGS().news() call to return the generated list.
        Verifies:
        1. The output contains at most 5 numbered articles (Requirement 5.2)
        2. The API is called with timelimit='w' to enforce 7-day window (Requirement 5.3)

        **Validates: Requirements 5.2, 5.3**
        """
        cache.clear()

        with patch("tools.market_scanner.DDGS") as mock_ddgs_class:
            mock_ddgs_instance = MagicMock()
            mock_ddgs_instance.news.return_value = news_items
            mock_ddgs_class.return_value = mock_ddgs_instance

            result = search_news("TEST")

        # Verify output contains at most 5 numbered articles
        import re
        numbered_lines = re.findall(r"^\d+\.", result, re.MULTILINE)
        assert len(numbered_lines) <= 5, (
            f"Expected at most 5 numbered articles, got {len(numbered_lines)} "
            f"from {len(news_items)} input items"
        )

        # Verify the API was called with timelimit="w" (7-day window)
        # This ensures all returned articles are from the most recent 7 days,
        # as the date filtering is delegated to the DuckDuckGo API.
        mock_ddgs_instance.news.assert_called_once_with(
            query="TEST", max_results=5, timelimit="w"
        )


class TestGetVolumeProperty:
    """Property-based tests for get_volume ratio computation and UNUSUAL VOLUME flag.

    Feature: agent-tools, Property 10: Volume ratio computation and UNUSUAL VOLUME flag

    **Validates: Requirements 7.2, 7.3**

    For any pair (current_volume, avg_volume) where avg_volume > 0, the volume ratio
    SHALL equal round(current_volume / avg_volume, 2), AND the string "UNUSUAL VOLUME"
    SHALL appear in the output if and only if the ratio exceeds 2.0.
    """

    def setup_method(self):
        """Clear cache before each test."""
        cache.clear()

    @given(
        current_volume=st.integers(min_value=1, max_value=10_000_000_000),
        prior_volumes=st.lists(
            st.integers(min_value=1, max_value=10_000_000_000),
            min_size=20,
            max_size=20,
        ),
    )
    @settings(max_examples=100)
    def test_volume_ratio_and_unusual_flag(self, current_volume, prior_volumes):
        """Volume ratio matches round(current_volume / avg_volume_float, 2) where
        avg_volume_float is the float mean of prior 20 volumes, and UNUSUAL VOLUME
        appears iff ratio > 2.0.

        **Validates: Requirements 7.2, 7.3**
        """
        cache.clear()

        # Build a DataFrame with 21 rows: 20 prior days + 1 current day
        volume_data = prior_volumes + [current_volume]
        hist_df = pd.DataFrame({"Volume": volume_data})

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist_df

        with patch("tools.market_scanner.yfinance.Ticker", return_value=mock_ticker):
            result = get_volume("TEST")

        # Compute expected ratio matching the implementation:
        # The implementation uses the float mean for ratio computation
        avg_volume_float = sum(prior_volumes) / len(prior_volumes)

        # If avg_volume_float is 0, the implementation returns an error
        if avg_volume_float == 0:
            assert "Error" in result
            return

        expected_ratio = round(current_volume / avg_volume_float, 2)

        # Verify ratio is present in output
        assert f"{expected_ratio}x" in result, (
            f"Expected ratio {expected_ratio}x not found in result: {result}"
        )

        # Verify UNUSUAL VOLUME flag presence iff ratio > 2.0
        if expected_ratio > 2.0:
            assert "UNUSUAL VOLUME" in result, (
                f"Expected 'UNUSUAL VOLUME' flag for ratio {expected_ratio} but not found in: {result}"
            )
        else:
            assert "UNUSUAL VOLUME" not in result, (
                f"Unexpected 'UNUSUAL VOLUME' flag for ratio {expected_ratio} in: {result}"
            )


class TestGetVolumeUnit:
    """Unit tests for get_volume function."""

    def setup_method(self):
        """Clear cache before each test."""
        cache.clear()

    def test_normal_volume_no_unusual_flag(self):
        """get_volume with ratio < 2.0 does NOT include UNUSUAL VOLUME flag."""
        # 20 prior days with volume 1,000,000 each, current day 1,500,000
        # Ratio = 1,500,000 / 1,000,000 = 1.5x (< 2.0, no flag)
        volume_data = [1_000_000] * 20 + [1_500_000]
        hist_df = pd.DataFrame({"Volume": volume_data})

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist_df

        with patch("tools.market_scanner.yfinance.Ticker", return_value=mock_ticker):
            result = get_volume("AAPL")

        assert "Volume Analysis for AAPL:" in result
        assert "Current Volume: 1,500,000" in result
        assert "20-Day Avg Volume: 1,000,000" in result
        assert "Volume Ratio: 1.5x" in result
        assert "UNUSUAL VOLUME" not in result

    def test_unusual_volume_flag_present(self):
        """get_volume with ratio > 2.0 includes UNUSUAL VOLUME flag."""
        # 20 prior days with volume 1,000,000 each, current day 3,000,000
        # Ratio = 3,000,000 / 1,000,000 = 3.0x (> 2.0, flag present)
        volume_data = [1_000_000] * 20 + [3_000_000]
        hist_df = pd.DataFrame({"Volume": volume_data})

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist_df

        with patch("tools.market_scanner.yfinance.Ticker", return_value=mock_ticker):
            result = get_volume("AAPL")

        assert "Volume Analysis for AAPL:" in result
        assert "Current Volume: 3,000,000" in result
        assert "20-Day Avg Volume: 1,000,000" in result
        assert "Volume Ratio: 3.0x" in result
        assert "UNUSUAL VOLUME" in result

    def test_invalid_ticker_returns_error(self):
        """get_volume returns error for empty ticker."""
        result = get_volume("")
        assert "Error" in result

    def test_ticker_not_found_returns_error(self):
        """get_volume returns error when ticker not found (empty history)."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()

        with patch("tools.market_scanner.yfinance.Ticker", return_value=mock_ticker):
            result = get_volume("XYZZY")

        assert "Error" in result
        assert "not found" in result

    def test_exception_returns_error_string(self):
        """get_volume catches exceptions and returns error string."""
        with patch("tools.market_scanner.yfinance.Ticker", side_effect=Exception("Network error")):
            result = get_volume("AAPL")

        assert "Error" in result
        assert "unexpected error" in result.lower()
