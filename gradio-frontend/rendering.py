"""HTML/CSS rendering for the FinAgent Gradio frontend.

Provides pure rendering functions that generate HTML strings for:
- Custom dark terminal theme CSS
- Trading signal cards (BUY/SELL/HOLD)
- Error cards for failed ticker analysis
- Aggregate summary bar
- Activity feed entries and container

Full implementation is provided in task 4.x; this module currently
contains function stubs.
"""

from datetime import datetime
from typing import Any, List, Optional


def build_css() -> str:
    """Generate custom CSS for the dark financial terminal theme.

    Returns:
        CSS string to be injected into the Gradio Blocks ``css`` parameter.
    """
    return """
    .gradio-container {
        font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
        background-color: #0d1117 !important;
    }
    .activity-feed {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 12px;
        max-height: 400px;
        overflow-y: auto;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
    }
    .activity-entry {
        padding: 4px 0;
        border-bottom: 1px solid #21262d;
        color: #c9d1d9;
    }
    .activity-timestamp {
        color: #8b949e;
        margin-right: 8px;
    }
    .activity-agent {
        color: #58a6ff;
        font-weight: bold;
    }
    .activity-spinner {
        color: #f0883e;
    }
    .signal-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        border-left: 4px solid;
    }
    .signal-buy { border-left-color: #3fb950; }
    .signal-sell { border-left-color: #f85149; }
    .signal-hold { border-left-color: #d29922; }
    .signal-error { border-left-color: #f85149; background: #1c0c0c; }
    .signal-ticker {
        font-size: 18px;
        font-weight: bold;
        color: #f0f6fc;
    }
    .signal-action-buy { color: #3fb950; }
    .signal-action-sell { color: #f85149; }
    .signal-action-hold { color: #d29922; }
    .signal-confidence {
        font-size: 14px;
        color: #8b949e;
    }
    .signal-prices {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        margin-top: 8px;
    }
    .signal-price-item {
        text-align: center;
        padding: 8px;
        background: #0d1117;
        border-radius: 4px;
    }
    .signal-price-label {
        font-size: 11px;
        color: #8b949e;
        text-transform: uppercase;
    }
    .signal-price-value {
        font-size: 16px;
        color: #f0f6fc;
        font-weight: bold;
    }
    .summary-bar {
        display: flex;
        gap: 16px;
        padding: 12px;
        background: #161b22;
        border-radius: 6px;
        margin-top: 12px;
        border: 1px solid #30363d;
    }
    .summary-item {
        text-align: center;
        flex: 1;
    }
    """


def _format_price(value: Optional[float]) -> str:
    """Format an optional price as ``$X.XX`` or ``N/A`` when missing."""
    if value is None:
        return "N/A"
    return f"${value:.2f}"


def _action_text(action: Any) -> str:
    """Return the upper-case action label from an enum-like or string value."""
    # Support both enum-like objects (with ``.value``) and plain strings.
    raw = getattr(action, "value", action)
    return str(raw).upper()


def render_signal_card(signal: Any) -> str:
    """Render a TradingSignal as an HTML card.

    Args:
        signal: TradingSignal dataclass instance containing ticker, action,
            confidence, entry/stop-loss/target prices, and reasoning.

    Returns:
        HTML string for the signal card with action-specific color coding.
    """
    action_text = _action_text(signal.action)
    action_lower = action_text.lower()
    action_class = f"signal-{action_lower}"
    action_color_class = f"signal-action-{action_lower}"

    entry_price = getattr(signal, "entry_price", None)
    stop_loss = getattr(signal, "stop_loss", None)
    target_price = getattr(signal, "target_price", None)

    prices_html = f"""
        <div class="signal-prices">
            <div class="signal-price-item">
                <div class="signal-price-label">Entry</div>
                <div class="signal-price-value">{_format_price(entry_price)}</div>
            </div>
            <div class="signal-price-item">
                <div class="signal-price-label">Stop Loss</div>
                <div class="signal-price-value">{_format_price(stop_loss)}</div>
            </div>
            <div class="signal-price-item">
                <div class="signal-price-label">Target</div>
                <div class="signal-price-value">{_format_price(target_price)}</div>
            </div>
        </div>
        """

    reasoning = getattr(signal, "reasoning", None) or {}
    reasoning_html = ""
    if reasoning:
        reasoning_items = "".join(
            f"<li><strong>{k}:</strong> {v}</li>"
            for k, v in reasoning.items()
        )
        reasoning_html = (
            f"<ul style='color:#c9d1d9;margin-top:8px;'>{reasoning_items}</ul>"
        )

    return f"""
    <div class="signal-card {action_class}">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span class="signal-ticker">{signal.ticker}</span>
            <span class="{action_color_class}" style="font-size:20px;font-weight:bold;">
                {action_text}
            </span>
        </div>
        <div class="signal-confidence">Confidence: {signal.confidence}%</div>
        {prices_html}
        {reasoning_html}
    </div>
    """


def render_error_card(ticker: str, error_message: str) -> str:
    """Render an error card for a failed ticker analysis.

    Args:
        ticker: The ticker symbol that failed analysis.
        error_message: Human-readable error description.

    Returns:
        HTML string for the error card with the ``signal-error`` class.
    """
    return f"""
    <div class="signal-card signal-error">
        <div class="signal-ticker">{ticker}</div>
        <div style="color:#f85149;margin-top:4px;">⚠️ Analysis failed: {error_message}</div>
    </div>
    """


def render_summary(
    total: int,
    buy_count: int,
    sell_count: int,
    hold_count: int,
) -> str:
    """Render the aggregate summary bar.

    Args:
        total: Total number of tickers analyzed.
        buy_count: Number of BUY signals.
        sell_count: Number of SELL signals.
        hold_count: Number of HOLD signals.

    Returns:
        HTML string for the summary bar with color-coded counts
        (green for BUY, red for SELL, yellow for HOLD).
    """
    return f"""
    <div class="summary-bar">
        <div class="summary-item">
            <div style="font-size:24px;color:#f0f6fc;">{total}</div>
            <div style="font-size:11px;color:#8b949e;">ANALYZED</div>
        </div>
        <div class="summary-item">
            <div style="font-size:24px;color:#3fb950;">{buy_count}</div>
            <div style="font-size:11px;color:#8b949e;">BUY</div>
        </div>
        <div class="summary-item">
            <div style="font-size:24px;color:#f85149;">{sell_count}</div>
            <div style="font-size:11px;color:#8b949e;">SELL</div>
        </div>
        <div class="summary-item">
            <div style="font-size:24px;color:#d29922;">{hold_count}</div>
            <div style="font-size:11px;color:#8b949e;">HOLD</div>
        </div>
    </div>
    """


def render_activity_entry(
    timestamp: datetime,
    agent_name: str,
    message: str,
    is_spinner: bool = False,
) -> str:
    """Render a single activity feed entry as HTML.

    Args:
        timestamp: Entry timestamp (formatted as HH:MM:SS).
        agent_name: Name of the agent producing the entry.
        message: Entry message content.
        is_spinner: When True, include a spinner indicator.

    Returns:
        HTML string for the activity entry.
    """
    time_str = timestamp.strftime("%H:%M:%S")
    spinner = '<span class="activity-spinner"> ⟳</span>' if is_spinner else ""
    return f"""
    <div class="activity-entry">
        <span class="activity-timestamp">[{time_str}]</span>
        <span class="activity-agent">{agent_name}</span>{spinner}
        <span>{message}</span>
    </div>
    """


def render_activity_feed(entries: List[str]) -> str:
    """Wrap activity entries in the feed container with auto-scroll.

    Args:
        entries: List of pre-rendered HTML entry strings.

    Returns:
        HTML string for the activity feed container, including an
        auto-scroll script that pins the view to the latest entry.
    """
    entries_html = "\n".join(entries)
    return f"""
    <div class="activity-feed" id="activity-feed">
        {entries_html}
    </div>
    <script>
        var feed = document.getElementById('activity-feed');
        if (feed) feed.scrollTop = feed.scrollHeight;
    </script>
    """
