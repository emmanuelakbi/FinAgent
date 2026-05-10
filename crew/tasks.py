"""Task factory functions with dependencies for CrewAI Task instances."""

from typing import Optional

from crewai import Task, Agent

from crew.config import TradePreferences


def create_market_scan_task(agent: Agent, ticker: str) -> Task:
    """Create the market scanning task."""
    return Task(
        description=(
            f"Analyze the current market conditions for {ticker}. "
            f"Search for recent news, check price changes, and identify volume anomalies. "
            f"Summarize any significant market events that could affect the stock."
        ),
        expected_output=(
            f"A summary of market conditions for {ticker} including: "
            f"key news events, price change magnitude and direction, "
            f"and whether volume is normal or unusual."
        ),
        agent=agent,
    )


def create_fundamental_task(agent: Agent, ticker: str) -> Task:
    """Create the fundamental analysis task."""
    return Task(
        description=(
            f"Perform a fundamental analysis of {ticker}. "
            f"Retrieve financial metrics, recent earnings data, and peer comparisons. "
            f"Assess whether the stock is overvalued, undervalued, or fairly valued."
        ),
        expected_output=(
            f"A valuation assessment for {ticker} including: "
            f"key financial metrics (P/E, margins, growth), "
            f"earnings trend and surprises, peer comparison, "
            f"and an overall fundamental outlook (bullish/bearish/neutral)."
        ),
        agent=agent,
    )


def create_technical_task(agent: Agent, ticker: str) -> Task:
    """Create the technical analysis task."""
    return Task(
        description=(
            f"Perform a technical analysis of {ticker}. "
            f"Retrieve price history and calculate technical indicators. "
            f"Identify the current trend, support/resistance levels, "
            f"and recommend entry and target prices."
        ),
        expected_output=(
            f"A technical analysis for {ticker} including: "
            f"current trend direction, RSI/MACD/Bollinger signals, "
            f"recommended entry price, and target price."
        ),
        agent=agent,
    )


def create_risk_task(
    agent: Agent,
    ticker: str,
    context: list,
    preferences: Optional[TradePreferences] = None,
) -> Task:
    """Create the risk assessment task.

    Args:
        agent: Risk Manager agent
        ticker: Stock symbol
        context: [technical_task] — depends on Technical Analyst output
        preferences: User preferences so the risk assessment matches the
            investor's profile (portfolio size, risk appetite).
    """
    prefs = preferences or TradePreferences()
    stop_target_pct = int(round(prefs.stop_pct * 100))
    target_target_pct = int(round(prefs.target_pct * 100))

    return Task(
        description=(
            f"Calculate position sizing and stop-loss levels for {ticker}. "
            f"Use the entry price from the Technical Analyst's recommendation "
            f"to determine optimal position size and ATR-based stop-loss.\n\n"
            f"The user's profile:\n"
            f"- Risk tolerance: {prefs.risk_tolerance} "
            f"(target stop-loss distance ~{stop_target_pct}% from entry)\n"
            f"- Trading style: {prefs.trading_style} "
            f"(target profit distance ~{target_target_pct}% from entry)\n"
            f"- Portfolio value: ${prefs.portfolio_value:,.0f}\n"
        ),
        expected_output=(
            f"Risk parameters for {ticker} including: "
            f"recommended position size, stop-loss price, "
            f"take-profit target, and risk-reward ratio."
        ),
        agent=agent,
        context=context,
    )


def create_strategy_task(
    agent: Agent,
    ticker: str,
    context: list,
    preferences: Optional[TradePreferences] = None,
) -> Task:
    r"""Create the strategy synthesis task.

    Args:
        agent: Chief Strategist agent
        ticker: Stock symbol
        context: [market_task, fundamental_task, technical_task, risk_task]
        preferences: User preferences that shape the final signal's
            stop / target distances and horizon. When ``None``, falls
            back to a Moderate / Swing Trading / \$10k profile.
    """
    prefs = preferences or TradePreferences()
    stop_pct = int(round(prefs.stop_pct * 100))
    target_pct = int(round(prefs.target_pct * 100))

    return Task(
        description=(
            f"Synthesize all analysis for {ticker} into a final trading "
            f"signal for a **{prefs.risk_tolerance} "
            f"{prefs.trading_style}** investor "
            f"with a ${prefs.portfolio_value:,.0f} portfolio.\n\n"
            f"Use the current price reported by get_price_change as Entry.\n"
            f"Profile-specific stop / target distances:\n"
            f"- Stop Loss: approximately {stop_pct}% from Entry "
            f"(tighter for Conservative, wider for Aggressive)\n"
            f"- Target: approximately {target_pct}% from Entry "
            f"(smaller for Day Trading, larger for Position Trading)\n\n"
            f"For BUY: Stop Loss below Entry, Target above Entry.\n"
            f"For SELL: Stop Loss above Entry, Target below Entry.\n"
            f"Keep the response concise — do not deliberate at length.\n\n"
            f"Output EXACTLY this format on its own lines, with NO extra prose:\n"
            f"{ticker} — BUY (Confidence: 75%)\n"
            f"Entry: $289.00\n"
            f"Stop Loss: $283.00\n"
            f"Target: $298.00\n"
            f"Reasoning:\n"
            f"- Market: [one line]\n"
            f"- Fundamental: [one line]\n"
            f"- Technical: [one line]\n"
            f"- Risk: [one line]\n\n"
            f"(Substitute the real action, confidence, prices, and reasoning; "
            f"keep every line on its own line; do not echo these instructions.)"
        ),
        expected_output=(
            f"A trading signal in the format: "
            f"'{ticker} — BUY/SELL/HOLD (Confidence: XX%)' "
            f"followed by entry, stop-loss, target prices and reasoning summaries."
        ),
        agent=agent,
        context=context,
    )
