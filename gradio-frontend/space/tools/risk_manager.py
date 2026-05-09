"""
Risk Manager Tools for FinAgent.

Provides tools for risk management agents:
- calculate_position_size: Calculate position size based on risk parameters
- set_stop_loss: Calculate ATR-based stop-loss and take-profit levels
"""

import math

import yfinance

# pandas-ta-remake is the maintained fork published under a different module
# name (pandas_ta_remake). Try it first, then fall back to the upstream
# pandas_ta name if it is what the environment provides.
try:
    import pandas_ta_remake as ta  # type: ignore
except ImportError:  # pragma: no cover - exercised only when remake is absent
    import pandas_ta as ta  # type: ignore

from crewai.tools import tool

from tools.cache import TTLCache
from tools.utils import validate_ticker

cache = TTLCache()


@tool("Calculate Position Size")
def calculate_position_size(
    portfolio_value: float,
    risk_percent: float,
    entry_price: float,
    stop_loss: float,
) -> str:
    """Calculate position size based on portfolio risk parameters.

    Args:
        portfolio_value: Total portfolio value in dollars.
        risk_percent: Percentage of portfolio to risk (0-100).
        entry_price: Planned entry price per share.
        stop_loss: Stop-loss price per share.

    Returns:
        Formatted string with position size details or error message.
    """
    try:
        # 1. Input validation
        if portfolio_value <= 0:
            return "Error: portfolio_value must be positive."

        if entry_price <= 0:
            return "Error: entry_price must be positive."

        if risk_percent <= 0 or risk_percent > 100:
            return "Error: risk_percent must be between 0 and 100."

        if entry_price == stop_loss:
            return "Error: entry_price and stop_loss cannot be equal."

        # 2. Calculate position size (no cache - pure computation)
        risk_amount = portfolio_value * risk_percent / 100
        risk_per_share = abs(entry_price - stop_loss)
        shares = math.floor(risk_amount / risk_per_share)
        total_position_value = shares * entry_price

        # 3. Format response
        response = (
            f"Position Size Calculation:\n"
            f"Portfolio Value: ${portfolio_value:,.2f}\n"
            f"Risk Amount: ${risk_amount:,.2f} ({risk_percent}%)\n"
            f"Entry Price: ${entry_price:,.2f}\n"
            f"Stop Loss: ${stop_loss:,.2f}\n"
            f"Risk Per Share: ${risk_per_share:,.2f}\n"
            f"Position Size: {shares} shares\n"
            f"Total Position Value: ${total_position_value:,.2f}"
        )

        return response

    except Exception as e:
        return f"Error: An unexpected error occurred: {str(e)}"


@tool("Set Stop Loss")
def set_stop_loss(ticker: str, entry_price: float, atr_multiplier: float) -> str:
    """Calculate ATR-based stop-loss and take-profit levels.

    Uses the 14-period Average True Range (ATR) to determine volatility-based
    stop-loss and take-profit levels for a given entry price.

    Args:
        ticker: Stock symbol (e.g., AAPL) or crypto pair (e.g., BTC-USD).
        entry_price: Entry price for the position.
        atr_multiplier: Multiplier for ATR (e.g., 1.5, 2.0).

    Returns:
        Formatted string with stop-loss, take-profit, and risk-reward ratio.
    """
    try:
        # 1. Input validation
        if atr_multiplier <= 0:
            return "Error: ATR multiplier must be positive."

        if entry_price <= 0:
            return "Error: Entry price must be positive."

        valid, result = validate_ticker(ticker)
        if not valid:
            return result

        normalized_ticker = result

        # 2. Cache check
        cache_key = cache.make_key(
            "set_stop_loss",
            ticker=normalized_ticker,
            entry_price=entry_price,
            atr_multiplier=atr_multiplier,
        )
        cached = cache.get(cache_key)
        if cached:
            return cached

        # 3. External API call - get price history for ATR calculation
        stock = yfinance.Ticker(normalized_ticker)
        df = stock.history(period="30d")

        if df.empty or len(df) < 14:
            return f"Error: Insufficient data to calculate ATR for {normalized_ticker}."

        # 4. Calculate ATR using pandas_ta
        atr_series = ta.atr(df["High"], df["Low"], df["Close"], length=14)

        if atr_series is None or atr_series.dropna().empty:
            return f"Error: Insufficient data to calculate ATR for {normalized_ticker}."

        atr_value = atr_series.dropna().iloc[-1]

        if atr_value != atr_value:  # Check for NaN
            return f"Error: Insufficient data to calculate ATR for {normalized_ticker}."

        # 5. Calculate stop-loss and take-profit
        stop_loss_price = round(entry_price - (atr_value * atr_multiplier), 2)
        take_profit_price = round(entry_price + (atr_value * atr_multiplier * 2), 2)

        # 6. Format response
        response = (
            f"Stop-Loss & Take-Profit for {normalized_ticker}:\n"
            f"Entry Price: ${entry_price:.2f}\n"
            f"ATR (14-period): ${atr_value:.2f}\n"
            f"ATR Multiplier: {atr_multiplier}x\n"
            f"Stop Loss: ${stop_loss_price:.2f} (Entry - ATR \u00d7 {atr_multiplier})\n"
            f"Take Profit: ${take_profit_price:.2f} (Entry + ATR \u00d7 {atr_multiplier * 2})\n"
            f"Risk/Reward Ratio: 1:2"
        )

        # 7. Cache and return
        cache.set(cache_key, response)
        return response

    except Exception as e:
        return f"Error: An unexpected error occurred while processing {ticker}: {str(e)}"
