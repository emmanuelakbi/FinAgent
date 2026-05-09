"""Task factory functions with dependencies for CrewAI Task instances."""

from crewai import Task, Agent


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


def create_risk_task(agent: Agent, ticker: str, context: list) -> Task:
    """Create the risk assessment task.

    Args:
        agent: Risk Manager agent
        ticker: Stock symbol
        context: [technical_task] — depends on Technical Analyst output
    """
    return Task(
        description=(
            f"Calculate position sizing and stop-loss levels for {ticker}. "
            f"Use the entry price from the Technical Analyst's recommendation "
            f"to determine optimal position size and ATR-based stop-loss."
        ),
        expected_output=(
            f"Risk parameters for {ticker} including: "
            f"recommended position size, stop-loss price, "
            f"take-profit target, and risk-reward ratio."
        ),
        agent=agent,
        context=context,
    )


def create_strategy_task(agent: Agent, ticker: str, context: list) -> Task:
    """Create the strategy synthesis task.

    Args:
        agent: Chief Strategist agent
        ticker: Stock symbol
        context: [market_task, fundamental_task, technical_task, risk_task]
    """
    return Task(
        description=(
            f"Synthesize all analysis for {ticker} into a final trading signal. "
            f"Consider the market conditions, fundamental valuation, technical signals, "
            f"and risk parameters. Produce a clear BUY, SELL, or HOLD recommendation "
            f"with a confidence percentage.\n\n"
            f"Your output MUST follow this exact format:\n"
            f"{ticker} — ACTION (Confidence: XX%)\n"
            f"Entry: $XX.XX\n"
            f"Stop Loss: $XX.XX\n"
            f"Target: $XX.XX\n"
            f"Reasoning:\n"
            f"- Market: [summary]\n"
            f"- Fundamental: [summary]\n"
            f"- Technical: [summary]\n"
            f"- Risk: [summary]"
        ),
        expected_output=(
            f"A trading signal in the format: "
            f"'{ticker} — BUY/SELL/HOLD (Confidence: XX%)' "
            f"followed by entry, stop-loss, target prices and reasoning summaries."
        ),
        agent=agent,
        context=context,
    )
