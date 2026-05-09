"""
Tests for cross-cutting error handling.

Feature: agent-tools
Property 1: Tool functions never raise exceptions

Covers:
- Tool functions never raise exceptions (Property 1)
- All 10 tool functions catch exceptions and return error strings

**Validates: Requirements 2.5, 2.1, 2.2, 2.3**
"""

from unittest.mock import patch, MagicMock

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from tools.market_scanner import search_news, get_price_change, get_volume
from tools.fundamental_analyst import get_financials, get_earnings, get_peers
from tools.technical_analyst import get_price_history, calculate_indicators
from tools.risk_manager import calculate_position_size, set_stop_loss
from tools.cache import TTLCache

# Strategy for generating random exception types
exception_types = st.sampled_from([
    RuntimeError,
    ConnectionError,
    TimeoutError,
    ValueError,
    KeyError,
    TypeError,
    AttributeError,
])

# Strategy for generating exception messages
exception_messages = st.text(min_size=0, max_size=50)

# Strategy for generating random exceptions
random_exceptions = st.builds(
    lambda exc_type, msg: exc_type(msg),
    exception_types,
    exception_messages,
)


def _clear_all_caches():
    """Clear caches in all tool modules to ensure fresh API calls."""
    from tools import market_scanner, fundamental_analyst, technical_analyst, risk_manager
    market_scanner.cache.clear()
    fundamental_analyst.cache.clear()
    technical_analyst.cache.clear()
    risk_manager.cache.clear()


class TestToolFunctionsNeverRaiseExceptions:
    """
    Feature: agent-tools
    Property 1: Tool functions never raise exceptions

    For any tool function and for any exception type raised by its internal logic
    (including network errors, invalid data, unexpected API responses, and runtime errors),
    the tool function SHALL catch the exception and return a string containing "Error"
    rather than propagating the exception to the caller.

    **Validates: Requirements 2.5, 2.1, 2.2, 2.3**
    """

    @given(exc=random_exceptions)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_search_news_never_raises(self, exc):
        """search_news catches all exceptions and returns error string."""
        _clear_all_caches()
        with patch("tools.market_scanner.DDGS") as mock_ddgs:
            mock_ddgs.return_value.news.side_effect = exc
            result = search_news("AAPL")
            assert isinstance(result, str)
            assert "Error" in result

    @given(exc=random_exceptions)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_get_price_change_never_raises(self, exc):
        """get_price_change catches all exceptions and returns error string."""
        _clear_all_caches()
        with patch("tools.market_scanner.yfinance.Ticker") as mock_ticker:
            mock_ticker.side_effect = exc
            result = get_price_change("AAPL")
            assert isinstance(result, str)
            assert "Error" in result

    @given(exc=random_exceptions)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_get_volume_never_raises(self, exc):
        """get_volume catches all exceptions and returns error string."""
        _clear_all_caches()
        with patch("tools.market_scanner.yfinance.Ticker") as mock_ticker:
            mock_ticker.side_effect = exc
            result = get_volume("AAPL")
            assert isinstance(result, str)
            assert "Error" in result

    @given(exc=random_exceptions)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_get_financials_never_raises(self, exc):
        """get_financials catches all exceptions and returns error string."""
        _clear_all_caches()
        with patch("tools.fundamental_analyst.yfinance.Ticker") as mock_ticker:
            mock_ticker.side_effect = exc
            result = get_financials("AAPL")
            assert isinstance(result, str)
            assert "Error" in result

    @given(exc=random_exceptions)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_get_earnings_never_raises(self, exc):
        """get_earnings catches all exceptions and returns error string."""
        _clear_all_caches()
        with patch("tools.fundamental_analyst.yfinance.Ticker") as mock_ticker:
            mock_ticker.side_effect = exc
            result = get_earnings("AAPL")
            assert isinstance(result, str)
            assert "Error" in result

    @given(exc=random_exceptions)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_get_peers_never_raises(self, exc):
        """get_peers catches all exceptions and returns error string."""
        _clear_all_caches()
        with patch("tools.fundamental_analyst.yfinance.Ticker") as mock_ticker:
            mock_ticker.side_effect = exc
            result = get_peers("AAPL")
            assert isinstance(result, str)
            assert "Error" in result

    @given(exc=random_exceptions)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_get_price_history_never_raises(self, exc):
        """get_price_history catches all exceptions and returns error string."""
        _clear_all_caches()
        with patch("tools.technical_analyst.yfinance.Ticker") as mock_ticker:
            mock_ticker.side_effect = exc
            result = get_price_history("AAPL")
            assert isinstance(result, str)
            assert "Error" in result

    @given(exc=random_exceptions)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_calculate_indicators_never_raises(self, exc):
        """calculate_indicators catches all exceptions and returns error string."""
        _clear_all_caches()
        with patch("tools.technical_analyst.yfinance.Ticker") as mock_ticker:
            mock_ticker.side_effect = exc
            result = calculate_indicators("AAPL")
            assert isinstance(result, str)
            assert "Error" in result

    @given(exc=random_exceptions)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_calculate_position_size_never_raises(self, exc):
        """calculate_position_size catches all exceptions and returns error string."""
        _clear_all_caches()
        # calculate_position_size is pure computation, so we mock math.floor
        # to simulate an internal failure
        with patch("tools.risk_manager.math.floor") as mock_floor:
            mock_floor.side_effect = exc
            result = calculate_position_size(
                portfolio_value=100000.0,
                risk_percent=1.0,
                entry_price=50.0,
                stop_loss=48.0,
            )
            assert isinstance(result, str)
            assert "Error" in result

    @given(exc=random_exceptions)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_set_stop_loss_never_raises(self, exc):
        """set_stop_loss catches all exceptions and returns error string."""
        _clear_all_caches()
        with patch("tools.risk_manager.yfinance.Ticker") as mock_ticker:
            mock_ticker.side_effect = exc
            result = set_stop_loss(
                ticker="AAPL",
                entry_price=185.0,
                atr_multiplier=1.5,
            )
            assert isinstance(result, str)
            assert "Error" in result
