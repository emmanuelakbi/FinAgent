"""Unit tests for the ``rendering`` module.

Complements the property-based tests in ``test_rendering_properties.py``
by asserting concrete example-based behaviour for each renderer:

* ``build_css`` exposes the dark financial-terminal palette and the
  monospace font stack.
* ``render_signal_card`` emits the correct action-specific CSS class and
  all user-facing fields for BUY / SELL / HOLD signals.
* ``render_error_card`` uses the ``signal-error`` class and surfaces the
  ticker and error message.
* ``render_summary`` renders each of the four count cells.
* ``render_activity_feed`` includes the auto-scroll script.

Requirements: 5.1, 5.2, 5.3, 5.5, 6.1, 6.4
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from rendering import (
    build_css,
    render_activity_feed,
    render_error_card,
    render_signal_card,
    render_summary,
)


# ---------------------------------------------------------------------------
# Test helpers: minimal stand-ins for the TradingSignal dataclass
# ---------------------------------------------------------------------------
#
# ``render_signal_card`` reads ``ticker``, ``action`` (enum-like with a
# ``.value`` attribute), ``confidence``, ``entry_price``, ``stop_loss``,
# ``target_price``, and ``reasoning`` off its argument. Declaring tiny
# local stand-ins keeps these unit tests hermetic — no need to import the
# real ``TradingSignal`` (and its downstream dependencies) from ``crew``.


class _FakeAction:
    """Stand-in for an enum member with a ``.value`` string attribute."""

    def __init__(self, value: str) -> None:
        self.value = value


@dataclass
class _FakeSignal:
    ticker: str
    action: _FakeAction
    confidence: int
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    reasoning: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# build_css
# ---------------------------------------------------------------------------


def test_build_css_contains_dark_colors():
    """CSS uses the GitHub-dark palette for the terminal backgrounds.

    Requirement 6.1 specifies a dark financial-terminal theme; the
    rendering module's design document fixes the background (#0d1117)
    and surface (#161b22) colours that identify the theme.
    """
    css = build_css()

    assert isinstance(css, str)
    assert "#0d1117" in css, (
        "Expected the primary dark background colour '#0d1117' in CSS"
    )
    assert "#161b22" in css, (
        "Expected the surface colour '#161b22' in CSS"
    )


def test_build_css_contains_monospace_font():
    """CSS declares the monospace font family required by Requirement 6.4."""
    css = build_css()

    assert "JetBrains Mono" in css, (
        "Expected 'JetBrains Mono' monospace font in CSS"
    )


# ---------------------------------------------------------------------------
# render_signal_card
# ---------------------------------------------------------------------------


def test_render_signal_card_buy():
    """BUY signal card exposes the green class and every user-facing field.

    Validates Requirements 5.1 (all fields displayed) and 5.2 (action
    colour coding — green for BUY).
    """
    signal = _FakeSignal(
        ticker="AAPL",
        action=_FakeAction("BUY"),
        confidence=85,
        entry_price=100.50,
        stop_loss=95.00,
        target_price=115.25,
        reasoning={"technical": "bullish breakout"},
    )

    html = render_signal_card(signal)

    # BUY-specific CSS class.
    assert "signal-buy" in html
    # Action label is uppercased.
    assert "BUY" in html
    # Ticker, confidence, and each price are rendered.
    assert "AAPL" in html
    assert "85%" in html
    assert "$100.50" in html
    assert "$95.00" in html
    assert "$115.25" in html
    # Reasoning map is rendered as a list item.
    assert "technical" in html
    assert "bullish breakout" in html


def test_render_signal_card_sell():
    """SELL signal card uses the red ``signal-sell`` class (Requirement 5.2)."""
    signal = _FakeSignal(
        ticker="TSLA",
        action=_FakeAction("SELL"),
        confidence=72,
        entry_price=250.00,
        stop_loss=260.00,
        target_price=230.00,
    )

    html = render_signal_card(signal)

    assert "signal-sell" in html
    assert "SELL" in html
    assert "TSLA" in html
    assert "72%" in html


def test_render_signal_card_hold():
    """HOLD signal card uses the yellow ``signal-hold`` class (Requirement 5.2)."""
    signal = _FakeSignal(
        ticker="MSFT",
        action=_FakeAction("HOLD"),
        confidence=60,
    )

    html = render_signal_card(signal)

    assert "signal-hold" in html
    assert "HOLD" in html
    assert "MSFT" in html
    assert "60%" in html
    # Missing prices fall back to ``N/A`` via ``_format_price``.
    assert "N/A" in html


# ---------------------------------------------------------------------------
# render_error_card
# ---------------------------------------------------------------------------


def test_render_error_card():
    """Error card uses the ``signal-error`` class and surfaces the message.

    Validates Requirement 5.5.
    """
    html = render_error_card("AAPL", "Connection timeout")

    assert "signal-error" in html
    assert "AAPL" in html
    assert "Connection timeout" in html


# ---------------------------------------------------------------------------
# render_summary
# ---------------------------------------------------------------------------


def test_render_summary():
    """Summary bar renders total / buy / sell / hold counts (Requirement 5.3).

    Assertions use the ``>{n}</div>`` pattern to scope each count to its
    summary cell, avoiding false positives from digit substrings (e.g. a
    ``2`` being satisfied by the ``2`` inside ``25``).
    """
    html = render_summary(total=5, buy_count=2, sell_count=1, hold_count=2)

    assert ">5</div>" in html  # total
    assert ">2</div>" in html  # buy count (also matches hold; both are 2)
    assert ">1</div>" in html  # sell count

    # Section labels are also present.
    assert "ANALYZED" in html
    assert "BUY" in html
    assert "SELL" in html
    assert "HOLD" in html


# ---------------------------------------------------------------------------
# render_activity_feed
# ---------------------------------------------------------------------------


def test_render_activity_feed_contains_scroll_script():
    """Feed container includes the auto-scroll script so the newest entry
    is visible without manual scrolling.
    """
    html = render_activity_feed([])

    # Auto-scroll script pins the view to the latest entry.
    assert "scrollTop" in html
    assert "scrollHeight" in html
    # Container has the id targeted by the script.
    assert 'id="activity-feed"' in html
