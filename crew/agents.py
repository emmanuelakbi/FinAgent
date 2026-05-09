"""Agent factory functions for creating configured CrewAI Agent instances."""

from crewai import Agent, LLM

from crew.config import LLMConfig


def create_llm(config: LLMConfig) -> LLM:
    """Create a shared crewai.LLM instance pointing to the vLLM endpoint.

    Newer versions of crewai validate ``Agent(llm=...)`` strictly and only
    accept a :class:`crewai.LLM` (or a plain model-name string). A raw
    ``langchain_openai.ChatOpenAI`` is rejected. crewai.LLM delegates to
    litellm under the hood, and litellm needs an explicit provider prefix
    for self-hosted OpenAI-compatible endpoints — hence the
    ``hosted_vllm/`` prefix on the model name.

    ``api_key`` is a required positional for the OpenAI provider flow even
    when the upstream server (vLLM) does not enforce auth; any non-empty
    value is accepted.
    """
    model_name = config.model_name
    if not model_name.startswith("hosted_vllm/"):
        model_name = f"hosted_vllm/{model_name}"

    return LLM(
        model=model_name,
        base_url=config.base_url,
        api_key="not-needed",
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.request_timeout,
    )


def create_market_scanner(llm: LLM, tools: list) -> Agent:
    """Create the Market Scanner agent.

    Args:
        llm: Shared LLM instance
        tools: [search_news, get_price_change, get_volume]
    """
    return Agent(
        role="Market Scanner",
        goal="Detect significant market events, price movements, and volume anomalies for the given ticker",
        backstory=(
            "You are an experienced market surveillance specialist who monitors "
            "news feeds, price action, and trading volumes 24/7. You have a keen "
            "eye for detecting material events that could impact stock prices."
        ),
        llm=llm,
        tools=tools,
        max_iter=5,
        verbose=True,
    )


def create_fundamental_analyst(llm: LLM, tools: list) -> Agent:
    """Create the Fundamental Analyst agent.

    Args:
        llm: Shared LLM instance
        tools: [get_financials, get_earnings, get_peers]
    """
    return Agent(
        role="Fundamental Analyst",
        goal="Determine the intrinsic value of the company by analyzing financial metrics, earnings trends, and peer comparisons",
        backstory=(
            "You are a seasoned equity research analyst with 15 years of experience "
            "in fundamental valuation. You specialize in dissecting financial statements, "
            "identifying earnings quality, and comparing companies against their peers."
        ),
        llm=llm,
        tools=tools,
        max_iter=5,
        verbose=True,
    )


def create_technical_analyst(llm: LLM, tools: list) -> Agent:
    """Create the Technical Analyst agent.

    Args:
        llm: Shared LLM instance
        tools: [get_price_history, calculate_indicators]
    """
    return Agent(
        role="Technical Analyst",
        goal="Identify optimal entry and exit points using price patterns and technical indicators",
        backstory=(
            "You are a quantitative technical analyst who combines classical chart "
            "patterns with modern indicator analysis. You focus on RSI, MACD, "
            "Bollinger Bands, and moving average crossovers to time entries precisely."
        ),
        llm=llm,
        tools=tools,
        max_iter=5,
        verbose=True,
    )


def create_risk_manager(llm: LLM, tools: list) -> Agent:
    """Create the Risk Manager agent.

    Args:
        llm: Shared LLM instance
        tools: [calculate_position_size, set_stop_loss]
    """
    return Agent(
        role="Risk Manager",
        goal="Protect capital through optimal position sizing and stop-loss placement based on volatility",
        backstory=(
            "You are a portfolio risk specialist who never lets a single trade "
            "risk more than the defined threshold. You use ATR-based stop-losses "
            "and position sizing formulas to ensure consistent risk management."
        ),
        llm=llm,
        tools=tools,
        max_iter=5,
        verbose=True,
    )


def create_chief_strategist(llm: LLM) -> Agent:
    """Create the Chief Strategist agent (no tools, pure reasoning)."""
    return Agent(
        role="Chief Strategist",
        goal="Synthesize all agent analyses into a single, actionable trading signal with confidence level",
        backstory=(
            "You are the head of trading strategy with decades of experience "
            "integrating fundamental, technical, and risk perspectives into "
            "decisive trading calls. You weigh conflicting signals and produce "
            "clear BUY/SELL/HOLD recommendations with calibrated confidence."
        ),
        llm=llm,
        tools=[],
        max_iter=5,
        verbose=True,
    )
