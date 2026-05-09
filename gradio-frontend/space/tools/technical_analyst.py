"""
Technical Analyst Tools for FinAgent.

Provides tools for technical analysis agents:
- get_price_history: Retrieve price history with calculated indicators
- calculate_indicators: Compute buy/sell signals from technical indicators
"""

# pandas-ta-remake is the maintained fork published under a different module
# name (pandas_ta_remake). Try it first, then fall back to the upstream
# pandas_ta name if it is what the environment provides.
try:
    import pandas_ta_remake as ta  # type: ignore
except ImportError:  # pragma: no cover - exercised only when remake is absent
    import pandas_ta as ta  # type: ignore

import yfinance
from crewai.tools import tool

from tools.cache import TTLCache
from tools.utils import validate_ticker

cache = TTLCache()


@tool("Get Price History")
def get_price_history(ticker: str) -> str:
    """Retrieve 60 days of price history with technical indicators including RSI, MACD, SMA, and Bollinger Bands.

    Args:
        ticker: Stock symbol (e.g., AAPL) or crypto pair (e.g., BTC-USD).

    Returns:
        A formatted table with the last 5 days of price data and indicators,
        or an error message if data is unavailable.
    """
    try:
        # 1. Input validation
        valid, result = validate_ticker(ticker)
        if not valid:
            return result

        normalized_ticker = result

        # 2. Cache check
        cache_key = cache.make_key("get_price_history", ticker=normalized_ticker)
        cached = cache.get(cache_key)
        if cached:
            return cached

        # 3. External API call — fetch 90 calendar days to ensure ~60 trading days
        stock = yfinance.Ticker(normalized_ticker)
        hist = stock.history(period="90d")

        if hist is None or hist.empty:
            return f"Error: Ticker '{normalized_ticker}' not found. Please verify the symbol."

        close = hist["Close"]
        num_days = len(close)

        # 4. Calculate indicators — mark as N/A if insufficient data
        insufficient_data = num_days < 50

        if insufficient_data:
            # Not enough data for SMA50; calculate what we can
            rsi = ta.rsi(close, length=14) if num_days >= 14 else None
            macd_df = ta.macd(close, fast=12, slow=26, signal=9) if num_days >= 26 else None
            sma20 = ta.sma(close, length=20) if num_days >= 20 else None
            sma50 = None
            bbands_df = ta.bbands(close, length=20, std=2) if num_days >= 20 else None
        else:
            rsi = ta.rsi(close, length=14)
            macd_df = ta.macd(close, fast=12, slow=26, signal=9)
            sma20 = ta.sma(close, length=20)
            sma50 = ta.sma(close, length=50)
            bbands_df = ta.bbands(close, length=20, std=2)

        # 5. Format response — last 5 days of computed data
        header = (
            f"Price History & Indicators for {normalized_ticker} (Last 5 Days):\n"
            f"Date       | Close   | RSI   | MACD  | Signal | SMA20   | SMA50   | BB_Upper | BB_Lower"
        )

        rows = []
        last_5_indices = hist.index[-5:]

        for idx in last_5_indices:
            date_str = idx.strftime("%Y-%m-%d")
            close_val = f"{close[idx]:.2f}"

            # RSI
            if rsi is not None and idx in rsi.index and not _is_na(rsi[idx]):
                rsi_val = f"{rsi[idx]:.1f}"
            else:
                rsi_val = "N/A"

            # MACD and Signal
            if macd_df is not None and idx in macd_df.index:
                macd_val_raw = macd_df["MACD_12_26_9"][idx]
                signal_val_raw = macd_df["MACDs_12_26_9"][idx]
                macd_val = f"{macd_val_raw:.2f}" if not _is_na(macd_val_raw) else "N/A"
                signal_val = f"{signal_val_raw:.2f}" if not _is_na(signal_val_raw) else "N/A"
            else:
                macd_val = "N/A"
                signal_val = "N/A"

            # SMA20
            if sma20 is not None and idx in sma20.index and not _is_na(sma20[idx]):
                sma20_val = f"{sma20[idx]:.2f}"
            else:
                sma20_val = "N/A"

            # SMA50
            if sma50 is not None and idx in sma50.index and not _is_na(sma50[idx]):
                sma50_val = f"{sma50[idx]:.2f}"
            else:
                sma50_val = "N/A"

            # Bollinger Bands
            if bbands_df is not None and idx in bbands_df.index:
                bbu_raw = bbands_df["BBU_20_2.0"][idx]
                bbl_raw = bbands_df["BBL_20_2.0"][idx]
                bbu_val = f"{bbu_raw:.2f}" if not _is_na(bbu_raw) else "N/A"
                bbl_val = f"{bbl_raw:.2f}" if not _is_na(bbl_raw) else "N/A"
            else:
                bbu_val = "N/A"
                bbl_val = "N/A"

            row = (
                f"{date_str} | {close_val:>7} | {rsi_val:>5} | {macd_val:>5} | "
                f"{signal_val:>6} | {sma20_val:>7} | {sma50_val:>7} | {bbu_val:>8} | {bbl_val:>8}"
            )
            rows.append(row)

        response = header + "\n" + "\n".join(rows)

        # 6. Cache and return
        cache.set(cache_key, response)
        return response

    except Exception as e:
        return f"Error: An unexpected error occurred while processing {ticker}: {str(e)}"


def _is_na(value) -> bool:
    """Check if a value is NaN or None."""
    if value is None:
        return True
    try:
        import math
        return math.isnan(value)
    except (TypeError, ValueError):
        return False


@tool("Calculate Indicators")
def calculate_indicators(ticker: str) -> str:
    """Compute current buy/sell signals from RSI, MACD, and Bollinger Bands for a given ticker.

    Args:
        ticker: Stock symbol (e.g., AAPL) or crypto pair (e.g., BTC-USD).

    Returns:
        A formatted string with each indicator's current value and signal classification
        (BUY, SELL, or NEUTRAL), or an error message if data is unavailable.
    """
    try:
        # 1. Input validation
        valid, result = validate_ticker(ticker)
        if not valid:
            return result

        normalized_ticker = result

        # 2. Cache check
        cache_key = cache.make_key("calculate_indicators", ticker=normalized_ticker)
        cached = cache.get(cache_key)
        if cached:
            return cached

        # 3. External API call — fetch 90 calendar days to ensure ~60 trading days
        stock = yfinance.Ticker(normalized_ticker)
        hist = stock.history(period="90d")

        if hist is None or hist.empty:
            return f"Error: Ticker '{normalized_ticker}' not found. Please verify the symbol."

        close = hist["Close"]
        num_days = len(close)

        if num_days < 26:
            return f"Error: Insufficient data for {normalized_ticker}. Need at least 26 trading days."

        # 4. Calculate indicators
        rsi_series = ta.rsi(close, length=14)
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        bbands_df = ta.bbands(close, length=20, std=2)

        # Get current values
        current_rsi = rsi_series.iloc[-1] if rsi_series is not None else None
        current_close = close.iloc[-1]

        # MACD values
        if macd_df is not None:
            current_macd = macd_df["MACD_12_26_9"].iloc[-1]
            current_signal = macd_df["MACDs_12_26_9"].iloc[-1]
            prev_macd = macd_df["MACD_12_26_9"].iloc[-2]
            prev_signal = macd_df["MACDs_12_26_9"].iloc[-2]
        else:
            current_macd = None
            current_signal = None
            prev_macd = None
            prev_signal = None

        # Bollinger Band values
        if bbands_df is not None:
            current_upper = bbands_df["BBU_20_2.0"].iloc[-1]
            current_lower = bbands_df["BBL_20_2.0"].iloc[-1]
        else:
            current_upper = None
            current_lower = None

        # 5. Classify signals
        # RSI classification
        if current_rsi is not None and not _is_na(current_rsi):
            if current_rsi < 30:
                rsi_signal = "BUY"
                rsi_desc = "Oversold"
            elif current_rsi > 70:
                rsi_signal = "SELL"
                rsi_desc = "Overbought"
            else:
                rsi_signal = "NEUTRAL"
                rsi_desc = "Neutral"
            rsi_line = f"RSI (14): {current_rsi:.1f} → {rsi_signal} ({rsi_desc})"
        else:
            rsi_line = "RSI (14): N/A → NEUTRAL (Insufficient Data)"

        # MACD classification
        if (current_macd is not None and current_signal is not None and
                prev_macd is not None and prev_signal is not None and
                not _is_na(current_macd) and not _is_na(current_signal) and
                not _is_na(prev_macd) and not _is_na(prev_signal)):
            is_bullish = current_macd > current_signal and prev_macd <= prev_signal
            is_bearish = current_macd < current_signal and prev_macd >= prev_signal

            if is_bullish:
                macd_signal = "BUY"
                macd_desc = "Bullish Crossover"
            elif is_bearish:
                macd_signal = "SELL"
                macd_desc = "Bearish Crossover"
            else:
                macd_signal = "NEUTRAL"
                macd_desc = "Neutral"
            macd_line = f"MACD: {current_macd:.2f} / Signal: {current_signal:.2f} → {macd_signal} ({macd_desc})"
        else:
            macd_line = "MACD: N/A / Signal: N/A → NEUTRAL (Insufficient Data)"

        # Bollinger classification
        if (current_upper is not None and current_lower is not None and
                not _is_na(current_upper) and not _is_na(current_lower)):
            if current_close < current_lower:
                bb_signal = "BUY"
                bb_desc = "Below Lower Band"
            elif current_close > current_upper:
                bb_signal = "SELL"
                bb_desc = "Above Upper Band"
            else:
                bb_signal = "NEUTRAL"
                bb_desc = "Neutral"
            bb_line = f"Bollinger: Price ${current_close:.2f} / Upper ${current_upper:.2f} / Lower ${current_lower:.2f} → {bb_signal} ({bb_desc})"
        else:
            bb_line = "Bollinger: N/A → NEUTRAL (Insufficient Data)"

        # 6. Format response
        response = (
            f"Technical Signals for {normalized_ticker}:\n"
            f"{rsi_line}\n"
            f"{macd_line}\n"
            f"{bb_line}"
        )

        # 7. Cache and return
        cache.set(cache_key, response)
        return response

    except Exception as e:
        return f"Error: An unexpected error occurred while processing {ticker}: {str(e)}"
