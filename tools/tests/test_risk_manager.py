"""
Tests for the Risk Manager tools.

Covers:
- Position size formula correctness (Property 14)
- Risk manager input validation (Property 15)
- Stop-loss and take-profit formula correctness (Property 16)
- Unit tests for specific scenarios
"""

from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

from tools.risk_manager import calculate_position_size, set_stop_loss, cache


class TestCalculatePositionSizeUnit:
    """Unit tests for calculate_position_size tool.

    Validates: Requirements 13.2
    """

    def setup_method(self):
        """Clear cache before each test."""
        cache.clear()

    def test_basic_position_size(self):
        """$100K portfolio, 1% risk, $50 entry, $48 stop → 500 shares."""
        result = calculate_position_size(
            portfolio_value=100000,
            risk_percent=1,
            entry_price=50,
            stop_loss=48,
        )

        # Verify shares: floor((100000 * 1 / 100) / |50 - 48|) = floor(1000 / 2) = 500
        assert "500 shares" in result
        # Verify risk amount: 100000 * 1 / 100 = $1,000
        assert "$1,000.00" in result
        # Verify total position value: 500 * 50 = $25,000
        assert "$25,000.00" in result
        # Verify no error
        assert "Error" not in result

    def test_position_size_output_contains_all_fields(self):
        """Verify the output contains all required fields."""
        result = calculate_position_size(
            portfolio_value=100000,
            risk_percent=1,
            entry_price=50,
            stop_loss=48,
        )

        assert "Portfolio Value:" in result
        assert "Risk Amount:" in result
        assert "Entry Price:" in result
        assert "Stop Loss:" in result
        assert "Risk Per Share:" in result
        assert "Position Size:" in result
        assert "Total Position Value:" in result


class TestSetStopLossUnit:
    """Unit tests for set_stop_loss tool.

    Validates: Requirements 14.3
    """

    def setup_method(self):
        """Clear cache before each test."""
        cache.clear()

    @patch("tools.risk_manager.ta.atr")
    @patch("tools.risk_manager.yfinance.Ticker")
    def test_basic_stop_loss(self, mock_ticker_class, mock_atr):
        """Entry $100, ATR $5, multiplier 1.5 → SL $92.50, TP $115.00."""
        # Mock yfinance history data (need at least 14 rows)
        dates = pd.date_range(end="2024-01-15", periods=20, freq="B")
        mock_df = pd.DataFrame(
            {
                "High": [105.0] * 20,
                "Low": [95.0] * 20,
                "Close": [100.0] * 20,
                "Open": [100.0] * 20,
                "Volume": [1000000] * 20,
            },
            index=dates,
        )

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_df
        mock_ticker_class.return_value = mock_ticker_instance

        # Mock pandas_ta.atr to return ATR=5.0
        atr_series = pd.Series([np.nan] * 13 + [5.0] * 7, index=dates)
        mock_atr.return_value = atr_series

        result = set_stop_loss(ticker="AAPL", entry_price=100, atr_multiplier=1.5)

        # Verify stop loss: 100 - (5.0 * 1.5) = 100 - 7.5 = $92.50
        assert "$92.50" in result
        # Verify take profit: 100 + (5.0 * 1.5 * 2) = 100 + 15.0 = $115.00
        assert "$115.00" in result
        # Verify risk/reward ratio
        assert "1:2" in result
        # Verify no error
        assert "Error" not in result

    @patch("tools.risk_manager.ta.atr")
    @patch("tools.risk_manager.yfinance.Ticker")
    def test_stop_loss_output_contains_all_fields(self, mock_ticker_class, mock_atr):
        """Verify the output contains all required fields."""
        dates = pd.date_range(end="2024-01-15", periods=20, freq="B")
        mock_df = pd.DataFrame(
            {
                "High": [105.0] * 20,
                "Low": [95.0] * 20,
                "Close": [100.0] * 20,
                "Open": [100.0] * 20,
                "Volume": [1000000] * 20,
            },
            index=dates,
        )

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_df
        mock_ticker_class.return_value = mock_ticker_instance

        atr_series = pd.Series([np.nan] * 13 + [5.0] * 7, index=dates)
        mock_atr.return_value = atr_series

        result = set_stop_loss(ticker="AAPL", entry_price=100, atr_multiplier=1.5)

        assert "Entry Price:" in result
        assert "ATR (14-period):" in result
        assert "ATR Multiplier:" in result
        assert "Stop Loss:" in result
        assert "Take Profit:" in result
        assert "Risk/Reward Ratio:" in result

import math

from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Property 14: Position size formula correctness
# Feature: agent-tools, Property 14: Position size formula correctness
# Validates: Requirements 13.1, 13.2
# ---------------------------------------------------------------------------


class TestPositionSizeFormulaCorrectness:
    """Property 14: Position size formula correctness

    For any valid inputs (portfolio_value > 0, 0 < risk_percent <= 100,
    entry_price > 0, stop_loss != entry_price), the computed number of shares
    SHALL equal floor((portfolio_value * risk_percent / 100) / abs(entry_price - stop_loss)),
    the dollar amount at risk SHALL equal portfolio_value * risk_percent / 100,
    and the total position value SHALL equal shares * entry_price.

    **Validates: Requirements 13.1, 13.2**
    """

    @given(
        portfolio_value=st.floats(min_value=1.0, max_value=1e8, allow_nan=False, allow_infinity=False),
        risk_percent=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
        entry_price=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
        stop_loss=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_shares_formula_correctness(self, portfolio_value, risk_percent, entry_price, stop_loss):
        """Verify shares = floor((portfolio_value * risk_percent / 100) / abs(entry_price - stop_loss)).

        Feature: agent-tools, Property 14: Position size formula correctness
        **Validates: Requirements 13.1, 13.2**
        """
        # Ensure entry_price != stop_loss
        assume(abs(entry_price - stop_loss) > 1e-9)

        result = calculate_position_size(
            portfolio_value=portfolio_value,
            risk_percent=risk_percent,
            entry_price=entry_price,
            stop_loss=stop_loss,
        )

        # Should not be an error
        assert "Error" not in result

        # Calculate expected values
        expected_risk_amount = portfolio_value * risk_percent / 100
        risk_per_share = abs(entry_price - stop_loss)
        expected_shares = math.floor(expected_risk_amount / risk_per_share)
        expected_total_value = expected_shares * entry_price

        # Verify shares count in output
        assert f"Position Size: {expected_shares} shares" in result

        # Verify total position value in output
        expected_total_formatted = f"${expected_total_value:,.2f}"
        assert f"Total Position Value: {expected_total_formatted}" in result

    @given(
        portfolio_value=st.floats(min_value=1.0, max_value=1e8, allow_nan=False, allow_infinity=False),
        risk_percent=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
        entry_price=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
        stop_loss=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_risk_amount_correctness(self, portfolio_value, risk_percent, entry_price, stop_loss):
        """Verify the dollar amount at risk equals portfolio_value * risk_percent / 100.

        Feature: agent-tools, Property 14: Position size formula correctness
        **Validates: Requirements 13.1, 13.2**
        """
        # Ensure entry_price != stop_loss
        assume(abs(entry_price - stop_loss) > 1e-9)

        result = calculate_position_size(
            portfolio_value=portfolio_value,
            risk_percent=risk_percent,
            entry_price=entry_price,
            stop_loss=stop_loss,
        )

        # Should not be an error
        assert "Error" not in result

        # Calculate expected risk amount
        expected_risk_amount = portfolio_value * risk_percent / 100
        expected_risk_formatted = f"${expected_risk_amount:,.2f}"

        # Verify risk amount in output
        assert expected_risk_formatted in result


# ---------------------------------------------------------------------------
# Property 16: Stop-loss and take-profit formula correctness
# Feature: agent-tools, Property 16: Stop-loss and take-profit formula correctness
# Validates: Requirements 14.2, 14.3
# ---------------------------------------------------------------------------


class TestStopLossTakeProfitFormulaCorrectness:
    """Feature: agent-tools, Property 16: Stop-loss and take-profit formula correctness

    For any valid (entry_price > 0, ATR > 0, atr_multiplier > 0), the stop-loss
    SHALL equal round(entry_price - (ATR * atr_multiplier), 2) and the take-profit
    SHALL equal round(entry_price + (ATR * atr_multiplier * 2), 2), yielding a
    risk-reward ratio of 1:2.

    **Validates: Requirements 14.2, 14.3**
    """

    @given(
        entry_price=st.floats(min_value=1.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
        atr_value=st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
        atr_multiplier=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_stop_loss_take_profit_formula_correctness(self, entry_price, atr_value, atr_multiplier):
        """
        Property 16: Stop-loss and take-profit formula correctness.

        For any valid (entry_price > 0, ATR > 0, atr_multiplier > 0), the stop-loss
        SHALL equal round(entry_price - (ATR * atr_multiplier), 2) and the take-profit
        SHALL equal round(entry_price + (ATR * atr_multiplier * 2), 2), yielding a
        risk-reward ratio of 1:2.

        **Validates: Requirements 14.2, 14.3**
        """
        # Clear cache before each test example
        from tools.cache import TTLCache
        from tools.risk_manager import cache as risk_cache
        risk_cache.clear()

        # Create mock price data with at least 14 rows for ATR calculation
        dates = pd.date_range(end="2024-01-15", periods=30, freq="B")
        mock_df = pd.DataFrame(
            {
                "Open": [entry_price] * 30,
                "High": [entry_price + 1.0] * 30,
                "Low": [entry_price - 1.0] * 30,
                "Close": [entry_price] * 30,
                "Volume": [1000000] * 30,
            },
            index=dates,
        )

        # Create a mock ATR series that returns our known ATR value
        atr_series = pd.Series([np.nan] * 13 + [atr_value] * 17, index=dates)

        with patch("tools.risk_manager.yfinance.Ticker") as mock_ticker_class, \
             patch("tools.risk_manager.ta.atr", return_value=atr_series):

            # Setup mocks
            mock_ticker_instance = MagicMock()
            mock_ticker_instance.history.return_value = mock_df
            mock_ticker_class.return_value = mock_ticker_instance

            result = set_stop_loss(
                ticker="AAPL",
                entry_price=entry_price,
                atr_multiplier=atr_multiplier,
            )

        # Verify the result is not an error
        assert "Error" not in result, f"Unexpected error: {result}"

        # Calculate expected values using numpy float64 to match the implementation
        # The implementation gets atr_value from a pandas Series (numpy.float64)
        # which has different rounding behavior than Python's native float
        np_atr_value = np.float64(atr_value)
        np_entry_price = np.float64(entry_price)
        np_multiplier = np.float64(atr_multiplier)

        expected_stop_loss = round(np_entry_price - (np_atr_value * np_multiplier), 2)
        expected_take_profit = round(np_entry_price + (np_atr_value * np_multiplier * 2), 2)

        # Format the same way the implementation does (using f-string with :.2f)
        expected_sl_str = f"${expected_stop_loss:.2f}"
        expected_tp_str = f"${expected_take_profit:.2f}"

        # Verify stop-loss formula: entry_price - (ATR * atr_multiplier)
        assert expected_sl_str in result, (
            f"Expected stop loss {expected_sl_str} not found in result.\n"
            f"entry_price={entry_price}, ATR={atr_value}, multiplier={atr_multiplier}\n"
            f"Result: {result}"
        )

        # Verify take-profit formula: entry_price + (ATR * atr_multiplier * 2)
        assert expected_tp_str in result, (
            f"Expected take profit {expected_tp_str} not found in result.\n"
            f"entry_price={entry_price}, ATR={atr_value}, multiplier={atr_multiplier}\n"
            f"Result: {result}"
        )

        # Verify risk-reward ratio is 1:2
        assert "1:2" in result, (
            f"Expected risk-reward ratio '1:2' not found in result.\n"
            f"Result: {result}"
        )

        # Verify the mathematical relationship: risk-reward = 1:2 (before rounding)
        # The formulas guarantee 1:2 ratio before rounding to 2 decimal places:
        # Risk (pre-round) = ATR * atr_multiplier
        # Reward (pre-round) = ATR * atr_multiplier * 2
        # Ratio = Reward / Risk = 2 (exactly)
        pre_round_risk = atr_value * atr_multiplier
        pre_round_reward = atr_value * atr_multiplier * 2
        assert abs(pre_round_reward - (2 * pre_round_risk)) < 1e-10, (
            f"Pre-rounding risk-reward ratio should be exactly 1:2.\n"
            f"Risk={pre_round_risk}, Reward={pre_round_reward}"
        )

class TestRiskManagerInputValidation:
    """Feature: agent-tools, Property 15: Risk manager input validation

    For any risk_percent < 0 or > 100, calculate_position_size SHALL return a string
    containing "Error". For any entry_price equal to stop_loss, calculate_position_size
    SHALL return "Error". For any portfolio_value <= 0 or entry_price <= 0,
    calculate_position_size SHALL return "Error". For any atr_multiplier <= 0,
    set_stop_loss SHALL return "Error".

    **Validates: Requirements 13.3, 13.4, 13.5, 14.4**
    """

    @given(
        risk_percent=st.one_of(
            st.floats(max_value=-0.01, min_value=-1e6, allow_nan=False, allow_infinity=False),
            st.floats(min_value=100.01, max_value=1e6, allow_nan=False, allow_infinity=False),
        ),
        portfolio_value=st.floats(min_value=1.0, max_value=1e9, allow_nan=False, allow_infinity=False),
        entry_price=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
        stop_loss=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_invalid_risk_percent_returns_error(
        self, risk_percent, portfolio_value, entry_price, stop_loss
    ):
        """risk_percent < 0 or > 100 SHALL return a string containing 'Error'.

        **Validates: Requirements 13.3**
        """
        assume(entry_price != stop_loss)
        result = calculate_position_size(
            portfolio_value=portfolio_value,
            risk_percent=risk_percent,
            entry_price=entry_price,
            stop_loss=stop_loss,
        )
        assert "Error" in result, (
            f"Expected 'Error' for risk_percent={risk_percent}, got: {result}"
        )

    @given(
        portfolio_value=st.floats(
            max_value=0.0, min_value=-1e9, allow_nan=False, allow_infinity=False
        ),
        risk_percent=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
        entry_price=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
        stop_loss=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_non_positive_portfolio_value_returns_error(
        self, portfolio_value, risk_percent, entry_price, stop_loss
    ):
        """portfolio_value <= 0 SHALL return a string containing 'Error'.

        **Validates: Requirements 13.5**
        """
        assume(entry_price != stop_loss)
        result = calculate_position_size(
            portfolio_value=portfolio_value,
            risk_percent=risk_percent,
            entry_price=entry_price,
            stop_loss=stop_loss,
        )
        assert "Error" in result, (
            f"Expected 'Error' for portfolio_value={portfolio_value}, got: {result}"
        )

    @given(
        entry_price=st.floats(
            max_value=0.0, min_value=-1e6, allow_nan=False, allow_infinity=False
        ),
        portfolio_value=st.floats(min_value=1.0, max_value=1e9, allow_nan=False, allow_infinity=False),
        risk_percent=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
        stop_loss=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_non_positive_entry_price_returns_error(
        self, entry_price, portfolio_value, risk_percent, stop_loss
    ):
        """entry_price <= 0 SHALL return a string containing 'Error'.

        **Validates: Requirements 13.5**
        """
        assume(entry_price != stop_loss)
        result = calculate_position_size(
            portfolio_value=portfolio_value,
            risk_percent=risk_percent,
            entry_price=entry_price,
            stop_loss=stop_loss,
        )
        assert "Error" in result, (
            f"Expected 'Error' for entry_price={entry_price}, got: {result}"
        )

    @given(
        price=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
        portfolio_value=st.floats(min_value=1.0, max_value=1e9, allow_nan=False, allow_infinity=False),
        risk_percent=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_equal_entry_and_stop_loss_returns_error(
        self, price, portfolio_value, risk_percent
    ):
        """entry_price == stop_loss SHALL return a string containing 'Error'.

        **Validates: Requirements 13.4**
        """
        result = calculate_position_size(
            portfolio_value=portfolio_value,
            risk_percent=risk_percent,
            entry_price=price,
            stop_loss=price,
        )
        assert "Error" in result, (
            f"Expected 'Error' when entry_price == stop_loss == {price}, got: {result}"
        )

    @given(
        atr_multiplier=st.floats(
            max_value=0.0, min_value=-1e6, allow_nan=False, allow_infinity=False
        ),
        entry_price=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
        ticker=st.just("AAPL"),
    )
    @settings(max_examples=100)
    def test_non_positive_atr_multiplier_returns_error(
        self, atr_multiplier, entry_price, ticker
    ):
        """atr_multiplier <= 0 SHALL return a string containing 'Error'.

        **Validates: Requirements 14.4**
        """
        result = set_stop_loss(
            ticker=ticker,
            entry_price=entry_price,
            atr_multiplier=atr_multiplier,
        )
        assert "Error" in result, (
            f"Expected 'Error' for atr_multiplier={atr_multiplier}, got: {result}"
        )
