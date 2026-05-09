"""
Fundamental Analyst Tools for FinAgent.

Provides tools for fundamental analysis agents:
- get_financials: Retrieve key financial metrics
- get_earnings: Retrieve earnings history with surprise calculations
- get_peers: Retrieve sector/industry peers
"""

import math

import yfinance
from crewai.tools import tool

from tools.cache import TTLCache
from tools.utils import validate_ticker, format_currency, format_percent, safe_get

cache = TTLCache()


@tool("Get Financials")
def get_financials(ticker: str) -> str:
    """Retrieve key financial metrics for a company including market cap, P/E ratio, revenue growth, profit margin, and debt-to-equity ratio."""
    try:
        # 1. Input validation
        valid, result = validate_ticker(ticker)
        if not valid:
            return result

        normalized_ticker = result

        # 2. Cache check
        cache_key = cache.make_key("get_financials", ticker=normalized_ticker)
        cached = cache.get(cache_key)
        if cached:
            return cached

        # 3. External API call
        info = yfinance.Ticker(normalized_ticker).info

        if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            return f"Error: Ticker '{normalized_ticker}' not found. Please verify the symbol."

        # 4. Format response — missing fields show "N/A", not an error
        market_cap = format_currency(info.get("marketCap"))
        pe_ratio = safe_get(info, "trailingPE")
        revenue_growth = format_percent(info.get("revenueGrowth"))
        profit_margin = format_percent(info.get("profitMargins"))
        debt_equity = safe_get(info, "debtToEquity")

        response = (
            f"Financial Metrics for {normalized_ticker}:\n"
            f"Market Cap: {market_cap}\n"
            f"P/E Ratio: {pe_ratio}\n"
            f"Revenue Growth: {revenue_growth}\n"
            f"Profit Margin: {profit_margin}\n"
            f"Debt/Equity: {debt_equity}"
        )

        # 5. Cache and return
        cache.set(cache_key, response)
        return response

    except Exception as e:
        return f"Error: An unexpected error occurred while processing {ticker}: {str(e)}"


@tool("Get Earnings")
def get_earnings(ticker: str) -> str:
    """Retrieve earnings history with surprise calculations for a given ticker.

    Args:
        ticker: Stock symbol (e.g., AAPL) or crypto pair (e.g., BTC-USD).

    Returns:
        A formatted string with the last 4 quarters of earnings data including
        reported EPS, estimated EPS, and surprise percentage, or an error message.
    """
    try:
        # 1. Input validation
        valid, result = validate_ticker(ticker)
        if not valid:
            return result

        normalized_ticker = result

        # 2. Detect crypto tickers (contain "-" like BTC-USD, ETH-USD)
        if "-" in normalized_ticker:
            return "Earnings data is not available for this instrument type."

        # 3. Cache check
        cache_key = cache.make_key("get_earnings", ticker=normalized_ticker)
        cached = cache.get(cache_key)
        if cached:
            return cached

        # 4. External API call
        stock = yfinance.Ticker(normalized_ticker)
        earnings_dates = stock.earnings_dates

        # 5. Check if earnings data is available
        if earnings_dates is None or earnings_dates.empty:
            return "Earnings data is not available for this instrument type."

        # 6. Filter to rows that have reported EPS (past earnings only)
        relevant_cols = ["EPS Estimate", "Reported EPS"]
        if not all(col in earnings_dates.columns for col in relevant_cols):
            return "Earnings data is not available for this instrument type."

        earnings_data = earnings_dates.dropna(subset=["Reported EPS"])

        if earnings_data.empty:
            return "Earnings data is not available for this instrument type."

        # Take the last 4 quarters (most recent first)
        earnings_data = earnings_data.head(4)

        # 7. Format response
        lines = [f"Earnings History for {normalized_ticker} (Last 4 Quarters):"]

        for date_idx, row in earnings_data.iterrows():
            reported_eps = row["Reported EPS"]
            estimated_eps = row.get("EPS Estimate")

            # Determine quarter label from the date index
            quarter_date = date_idx
            quarter_num = (quarter_date.month - 1) // 3 + 1
            quarter_label = f"Q{quarter_num} {quarter_date.year}"

            # Calculate surprise percentage
            if estimated_eps is not None and estimated_eps != 0:
                if not math.isnan(estimated_eps):
                    surprise = round(((reported_eps - estimated_eps) / abs(estimated_eps)) * 100, 2)
                    surprise_str = f"+{surprise:.2f}%" if surprise >= 0 else f"{surprise:.2f}%"
                    lines.append(
                        f"{quarter_label}: EPS ${reported_eps:.2f} "
                        f"(Est: ${estimated_eps:.2f}) | Surprise: {surprise_str}"
                    )
                else:
                    lines.append(
                        f"{quarter_label}: EPS ${reported_eps:.2f} "
                        f"(Est: N/A) | Surprise: N/A"
                    )
            else:
                lines.append(
                    f"{quarter_label}: EPS ${reported_eps:.2f} "
                    f"(Est: N/A) | Surprise: N/A"
                )

        response = "\n".join(lines)

        # 8. Cache and return
        cache.set(cache_key, response)
        return response

    except Exception as e:
        return f"Error: An unexpected error occurred while processing {ticker}: {str(e)}"


# Sector-to-peers mapping for common sectors
SECTOR_PEERS = {
    "Technology": [
        ("MSFT", "Microsoft Corporation"),
        ("AAPL", "Apple Inc."),
        ("GOOGL", "Alphabet Inc."),
        ("NVDA", "NVIDIA Corporation"),
        ("META", "Meta Platforms Inc."),
        ("AMZN", "Amazon.com Inc."),
        ("CRM", "Salesforce Inc."),
        ("ADBE", "Adobe Inc."),
        ("ORCL", "Oracle Corporation"),
        ("INTC", "Intel Corporation"),
    ],
    "Healthcare": [
        ("JNJ", "Johnson & Johnson"),
        ("UNH", "UnitedHealth Group Inc."),
        ("PFE", "Pfizer Inc."),
        ("ABBV", "AbbVie Inc."),
        ("MRK", "Merck & Co. Inc."),
        ("LLY", "Eli Lilly and Company"),
        ("TMO", "Thermo Fisher Scientific Inc."),
        ("ABT", "Abbott Laboratories"),
    ],
    "Financial Services": [
        ("JPM", "JPMorgan Chase & Co."),
        ("BAC", "Bank of America Corporation"),
        ("GS", "Goldman Sachs Group Inc."),
        ("MS", "Morgan Stanley"),
        ("WFC", "Wells Fargo & Company"),
        ("C", "Citigroup Inc."),
        ("BLK", "BlackRock Inc."),
        ("SCHW", "Charles Schwab Corporation"),
    ],
    "Consumer Cyclical": [
        ("AMZN", "Amazon.com Inc."),
        ("TSLA", "Tesla Inc."),
        ("HD", "The Home Depot Inc."),
        ("NKE", "Nike Inc."),
        ("MCD", "McDonald's Corporation"),
        ("SBUX", "Starbucks Corporation"),
        ("TGT", "Target Corporation"),
        ("LOW", "Lowe's Companies Inc."),
    ],
    "Consumer Defensive": [
        ("PG", "Procter & Gamble Company"),
        ("KO", "The Coca-Cola Company"),
        ("PEP", "PepsiCo Inc."),
        ("WMT", "Walmart Inc."),
        ("COST", "Costco Wholesale Corporation"),
        ("CL", "Colgate-Palmolive Company"),
        ("MDLZ", "Mondelez International Inc."),
    ],
    "Communication Services": [
        ("GOOGL", "Alphabet Inc."),
        ("META", "Meta Platforms Inc."),
        ("DIS", "The Walt Disney Company"),
        ("NFLX", "Netflix Inc."),
        ("CMCSA", "Comcast Corporation"),
        ("T", "AT&T Inc."),
        ("VZ", "Verizon Communications Inc."),
    ],
    "Industrials": [
        ("CAT", "Caterpillar Inc."),
        ("HON", "Honeywell International Inc."),
        ("UPS", "United Parcel Service Inc."),
        ("BA", "The Boeing Company"),
        ("GE", "General Electric Company"),
        ("RTX", "RTX Corporation"),
        ("DE", "Deere & Company"),
        ("LMT", "Lockheed Martin Corporation"),
    ],
    "Energy": [
        ("XOM", "Exxon Mobil Corporation"),
        ("CVX", "Chevron Corporation"),
        ("COP", "ConocoPhillips"),
        ("SLB", "Schlumberger Limited"),
        ("EOG", "EOG Resources Inc."),
        ("OXY", "Occidental Petroleum Corporation"),
        ("MPC", "Marathon Petroleum Corporation"),
    ],
    "Real Estate": [
        ("AMT", "American Tower Corporation"),
        ("PLD", "Prologis Inc."),
        ("CCI", "Crown Castle Inc."),
        ("EQIX", "Equinix Inc."),
        ("SPG", "Simon Property Group Inc."),
        ("O", "Realty Income Corporation"),
    ],
    "Utilities": [
        ("NEE", "NextEra Energy Inc."),
        ("DUK", "Duke Energy Corporation"),
        ("SO", "The Southern Company"),
        ("D", "Dominion Energy Inc."),
        ("AEP", "American Electric Power Company Inc."),
        ("SRE", "Sempra"),
    ],
    "Basic Materials": [
        ("LIN", "Linde plc"),
        ("APD", "Air Products and Chemicals Inc."),
        ("SHW", "The Sherwin-Williams Company"),
        ("FCX", "Freeport-McMoRan Inc."),
        ("NEM", "Newmont Corporation"),
        ("DOW", "Dow Inc."),
    ],
}


@tool("Get Peers")
def get_peers(ticker: str) -> str:
    """Retrieve sector/industry classification and peer companies for a given ticker.
    Returns sector, industry, and up to 5 peer companies in the same sector.
    Use this to contextualize a company's performance relative to competitors."""
    try:
        # 1. Input validation
        valid, result = validate_ticker(ticker)
        if not valid:
            return result

        normalized_ticker = result

        # 2. Cache check
        cache_key = cache.make_key("get_peers", ticker=normalized_ticker)
        cached = cache.get(cache_key)
        if cached:
            return cached

        # 3. Detect crypto/ETFs early by ticker pattern (e.g., BTC-USD, ETH-USD)
        if "-" in normalized_ticker:
            return "Peer comparison is not available for this instrument type."

        # 4. External API call
        yf_ticker = yfinance.Ticker(normalized_ticker)
        info = yf_ticker.info

        # 5. Check if sector is available (missing for crypto/ETFs)
        sector = info.get("sector")
        industry = info.get("industry")

        if not sector:
            return "Peer comparison is not available for this instrument type."

        # 6. Identify peers from sector mapping, excluding the ticker itself
        sector_companies = SECTOR_PEERS.get(sector, [])
        peers = [
            (sym, name)
            for sym, name in sector_companies
            if sym != normalized_ticker
        ][:5]

        # 7. Format response
        lines = [f"Peer Analysis for {normalized_ticker}:"]
        lines.append(f"Sector: {sector}")
        lines.append(f"Industry: {industry if industry else 'N/A'}")

        if peers:
            lines.append("Peers:")
            for sym, name in peers:
                lines.append(f"- {sym} ({name})")
        else:
            lines.append("Peers: No peer data available for this sector.")

        response = "\n".join(lines)

        # 8. Cache and return
        cache.set(cache_key, response)
        return response

    except Exception as e:
        return f"Error: An unexpected error occurred while processing {ticker}: {str(e)}"
