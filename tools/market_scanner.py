"""
Market Scanner Tools for FinAgent.

Provides tools for market scanning agents:
- search_news: Search recent news for a ticker
- get_price_change: Get current price vs previous close
- get_volume: Get volume analysis with unusual activity detection
"""

import yfinance
from crewai.tools import tool
from ddgs import DDGS

from tools.cache import TTLCache
from tools.utils import validate_ticker

cache = TTLCache()


@tool("Search News")
def search_news(ticker: str) -> str:
    """Search recent news articles for a given ticker symbol.

    Args:
        ticker: Stock symbol (e.g., AAPL) or crypto pair (e.g., BTC-USD).

    Returns:
        A formatted string with up to 5 recent news articles including
        title and snippet, or an error message if something goes wrong.
    """
    try:
        # 1. Input validation
        valid, result = validate_ticker(ticker)
        if not valid:
            return result

        normalized_ticker = result

        # 2. Cache check
        cache_key = cache.make_key("search_news", ticker=normalized_ticker)
        cached = cache.get(cache_key)
        if cached:
            return cached

        # 3. External API call
        ddgs = DDGS()
        articles = ddgs.news(query=normalized_ticker, max_results=5, timelimit="w")

        # 4. Format response
        if not articles:
            response = f"No recent news found for {normalized_ticker} in the last 7 days."
        else:
            lines = [f"Recent News for {normalized_ticker} (last 7 days):"]
            for i, article in enumerate(articles[:5], start=1):
                title = article.get("title", "No title")
                snippet = article.get("body", "No description available")
                lines.append(f"{i}. {title} - {snippet}")
            response = "\n".join(lines)

        # 5. Cache and return
        cache.set(cache_key, response)
        return response

    except Exception as e:
        return f"Error: An unexpected error occurred while processing {ticker}: {str(e)}"


@tool("Get Price Change")
def get_price_change(ticker: str) -> str:
    """Get current price vs previous close and calculate the change for a ticker.

    Retrieves the current price and previous closing price, then calculates
    the absolute and percentage change.

    Args:
        ticker: Stock symbol (e.g., "AAPL") or crypto pair (e.g., "BTC-USD").

    Returns:
        Formatted string with price change data or error message.
    """
    try:
        # 1. Input validation
        valid, result = validate_ticker(ticker)
        if not valid:
            return result

        normalized_ticker = result

        # 2. Cache check
        cache_key = cache.make_key("get_price_change", ticker=normalized_ticker)
        cached = cache.get(cache_key)
        if cached:
            return cached

        # 3. External API call
        stock = yfinance.Ticker(normalized_ticker)
        info = stock.info

        current_price = info.get("currentPrice")
        previous_close = info.get("previousClose")

        # Crypto tickers (e.g. BTC-USD) expose regularMarketPrice rather than
        # currentPrice. Fall back to it when currentPrice is missing.
        if current_price is None:
            current_price = info.get("regularMarketPrice")

        # Fall back to history when either value is still unavailable.
        # Some tickers (notably crypto) only return a single row for a 2-day
        # lookback because trading is continuous, so also try a longer window.
        if current_price is None or previous_close is None:
            hist = stock.history(period="2d")
            if hist is None or hist.empty:
                return f"Error: Ticker '{normalized_ticker}' not found. Please verify the symbol."
            if len(hist) < 2:
                # Retry with a wider window to recover a previous close.
                hist = stock.history(period="5d")
                if hist is None or len(hist) < 2:
                    return (
                        f"Error: Ticker '{normalized_ticker}' not found. "
                        f"Please verify the symbol."
                    )
            previous_close = float(hist["Close"].iloc[-2])
            current_price = float(hist["Close"].iloc[-1])

        # 4. Data validation
        if previous_close is None or previous_close == 0:
            return f"Error: Required data unavailable for {normalized_ticker}: previousClose"

        if current_price is None:
            return f"Error: Required data unavailable for {normalized_ticker}: currentPrice"

        current_price = float(current_price)
        previous_close = float(previous_close)

        # 5. Calculate change
        absolute_change = current_price - previous_close
        percent_change = round(((current_price - previous_close) / previous_close) * 100, 2)

        # 6. Format response
        sign = "+" if absolute_change >= 0 else "-"
        abs_change = abs(absolute_change)

        response = (
            f"Price Change for {normalized_ticker}:\n"
            f"Current Price: ${current_price:.2f}\n"
            f"Previous Close: ${previous_close:.2f}\n"
            f"Change: {sign}${abs_change:.2f} ({'+' if percent_change >= 0 else ''}{percent_change}%)"
        )

        # 7. Cache and return
        cache.set(cache_key, response)
        return response

    except Exception as e:
        return f"Error: An unexpected error occurred while processing {ticker}: {str(e)}"


@tool("Get Volume Analysis")
def get_volume(ticker: str) -> str:
    """Get volume analysis with unusual activity detection for a ticker.

    Retrieves current volume and 20-day average volume, calculates the
    volume ratio, and flags unusual activity when ratio exceeds 2.0.

    Args:
        ticker: Stock symbol (e.g., "AAPL") or crypto pair (e.g., "BTC-USD").

    Returns:
        Formatted string with volume analysis or error message.
    """
    try:
        # 1. Input validation
        valid, result = validate_ticker(ticker)
        if not valid:
            return result

        normalized_ticker = result

        # 2. Cache check
        cache_key = cache.make_key("get_volume", ticker=normalized_ticker)
        cached = cache.get(cache_key)
        if cached:
            return cached

        # 3. External API call
        stock = yfinance.Ticker(normalized_ticker)
        hist = stock.history(period="25d")

        if hist.empty or len(hist) < 2:
            return f"Error: Ticker '{normalized_ticker}' not found. Please verify the symbol."

        if "Volume" not in hist.columns:
            return f"Error: Required data unavailable for {normalized_ticker}: Volume"

        # 4. Data validation and computation
        current_volume = int(hist["Volume"].iloc[-1])
        # Average of prior 20 days (excluding the most recent day)
        prior_volumes = hist["Volume"].iloc[:-1]
        # Take up to 20 days for the average
        prior_20 = prior_volumes.tail(20)

        if len(prior_20) == 0 or prior_20.mean() == 0:
            return f"Error: Required data unavailable for {normalized_ticker}: insufficient volume history"

        avg_volume_float = prior_20.mean()
        avg_volume = int(avg_volume_float)
        volume_ratio = round(current_volume / avg_volume_float, 2)

        # 5. Format response
        response = (
            f"Volume Analysis for {normalized_ticker}:\n"
            f"Current Volume: {current_volume:,}\n"
            f"20-Day Avg Volume: {avg_volume:,}\n"
            f"Volume Ratio: {volume_ratio}x"
        )

        # Include UNUSUAL VOLUME flag when ratio > 2.0
        if volume_ratio > 2.0:
            response += "\n⚠️ UNUSUAL VOLUME"

        # 6. Cache and return
        cache.set(cache_key, response)
        return response

    except Exception as e:
        return f"Error: An unexpected error occurred while processing {ticker}: {str(e)}"
