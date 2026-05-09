"""Integration tests for ``app.run_analysis``.

These tests drive the full Analyze-button generator end-to-end with a
fake ``crew`` package injected via ``sys.modules``, verifying the
streaming contract the real UI depends on:

* The button starts the run disabled (with progress visible) and ends
  the run re-enabled (with progress hidden), on both the happy path
  and partial-failure paths.
* Each ticker contributes either a signal card or an error card to the
  final dashboard HTML, preserving the summary-bar + cards layout.
* Callback events emitted by the runner are translated faithfully into
  the activity feed HTML — including the ``⟳`` spinner glyph for
  ``task_start`` events, per Requirement 3.2.

All fakes come from ``tests/conftest.py`` (``buy_signal``,
``sell_signal``, ``FakeCrewResult``, ``make_fake_crew_package``,
``sample_activity_events``), so these tests never touch the real
``crewai`` runtime and can run in any environment where ``gradio`` is
installed.

Yields from ``run_analysis`` are 7-tuples in the order wired up by
``analyze_btn.click``'s ``outputs=``::

    (analyze_btn, error_display, progress_text,
     activity_feed, signals_dashboard,
     activity_log, signals_state)

``gr.update(...)`` returns a plain ``dict`` in gradio 4.44.x, so fields
are asserted via ``dict.get(...)``.

Validates: Requirements 3.1, 3.3, 5.1, 5.5, 8.1.
"""

from __future__ import annotations

import sys
from typing import List
from unittest.mock import patch

import pytest

# Skip the whole module cleanly when gradio isn't installed — the handler
# imports gradio at module import time.
pytest.importorskip("gradio")

import app as app_module  # noqa: E402  (import deferred until after importorskip)

# NOTE: The helpers live in ``tests/conftest.py`` alongside the fixtures.
# A separate root-level ``conftest.py`` at the repository top (see
# ``FinAgent/conftest.py``) shadows a bare ``from conftest import ...``,
# so import via the ``tests`` package path instead — ``tests/__init__.py``
# makes this a regular package and the path stays valid regardless of the
# active pytest rootdir.
from tests.conftest import (  # noqa: E402  (local test helpers)
    FakeActivityEvent,
    FakeCrewResult,
    FakeSignal,
    make_fake_crew_package,
)


# ---------------------------------------------------------------------------
# Yield-tuple layout — mirrored from ``test_handler.py`` for consistency.
# ---------------------------------------------------------------------------

_YIELD_TUPLE_LEN = 7
_IDX_ANALYZE_BTN = 0
_IDX_ERROR_DISPLAY = 1
_IDX_PROGRESS_TEXT = 2
_IDX_ACTIVITY_FEED = 3
_IDX_SIGNALS_DASHBOARD = 4
_IDX_ACTIVITY_LOG = 5
_IDX_SIGNALS_STATE = 6


def _drive_run_analysis(
    ticker_input: str,
    fake_crew,
    fake_callbacks,
    *,
    risk_tolerance: str = "Moderate",
    portfolio_value: float = 10000,
    trading_style: str = "Swing Trading",
) -> list:
    """Run ``run_analysis`` against injected fake crew modules and collect yields.

    Wraps the standard ``patch.object`` / ``patch.dict`` incantation so
    each test reads cleanly as "build signals map, then drive the
    handler and inspect yields".
    """
    with patch.object(app_module, "VLLM_ENDPOINT_URL", "http://test-endpoint"), \
         patch.dict(
             sys.modules,
             {"crew": fake_crew, "crew.callbacks": fake_callbacks},
         ):
        return list(
            app_module.run_analysis(
                ticker_input=ticker_input,
                risk_tolerance=risk_tolerance,
                portfolio_value=portfolio_value,
                trading_style=trading_style,
                activity_log=[],
                signals_state=[],
            )
        )


# ---------------------------------------------------------------------------
# Test 1: Full generator sequence for two successful tickers
# ---------------------------------------------------------------------------


def test_full_generator_sequence_two_tickers(
    buy_signal: FakeSignal,
    sell_signal: FakeSignal,
) -> None:
    """Two-ticker happy path streams: disabled → progress → cards → re-enabled.

    Drives ``run_analysis`` with a fake crew that returns a BUY signal
    for AAPL and a SELL signal for TSLA. Asserts the generator emits at
    least three yields (initial + one per ticker), that the initial and
    intermediate yields keep the Analyze button disabled while showing
    progress, and that the final yield re-enables the button, hides
    progress, and renders both signal cards with their color-coded CSS
    classes.

    Validates: Requirements 3.1, 5.1, 8.1.
    """
    signals = {
        "AAPL": FakeCrewResult(success=True, signal=buy_signal),
        "TSLA": FakeCrewResult(success=True, signal=sell_signal),
    }
    fake_crew, fake_callbacks = make_fake_crew_package(signals)

    results = _drive_run_analysis("AAPL, TSLA", fake_crew, fake_callbacks)

    # Initial yield + one yield per ticker = at least 3.
    assert len(results) >= 3, (
        f"Expected at least 3 yields (initial + 2 ticker updates), "
        f"got {len(results)}: {results!r}"
    )
    for i, y in enumerate(results):
        assert len(y) == _YIELD_TUPLE_LEN, (
            f"yield #{i} has wrong arity {len(y)}: {y!r}"
        )

    # --- First yield: button disabled, progress visible ---
    first = results[0]
    assert first[_IDX_ANALYZE_BTN].get("interactive") is False, (
        f"first yield must disable the button, got: {first[_IDX_ANALYZE_BTN]!r}"
    )
    assert first[_IDX_PROGRESS_TEXT].get("visible") is True, (
        f"first yield must show progress, got: {first[_IDX_PROGRESS_TEXT]!r}"
    )

    # --- Intermediate yields: button stays disabled while analysis runs ---
    # Every yield except the last represents mid-run state and must keep
    # the button disabled so the user can't double-submit.
    for idx, y in enumerate(results[:-1]):
        btn = y[_IDX_ANALYZE_BTN]
        assert btn.get("interactive") is False, (
            f"mid-run yield #{idx} must keep the button disabled, got: {btn!r}"
        )

    # --- Final yield: button re-enabled, progress hidden ---
    final = results[-1]
    assert final[_IDX_ANALYZE_BTN].get("interactive") is True, (
        f"final yield must re-enable the button, got: {final[_IDX_ANALYZE_BTN]!r}"
    )
    assert final[_IDX_PROGRESS_TEXT].get("visible") is False, (
        f"final yield must hide progress, got: {final[_IDX_PROGRESS_TEXT]!r}"
    )

    # --- Final dashboard: both tickers present with color-coded classes ---
    dashboard = final[_IDX_SIGNALS_DASHBOARD]
    assert isinstance(dashboard, str)
    assert "AAPL" in dashboard, (
        f"dashboard missing AAPL ticker, got: {dashboard!r}"
    )
    assert "TSLA" in dashboard, (
        f"dashboard missing TSLA ticker, got: {dashboard!r}"
    )
    assert "signal-buy" in dashboard, (
        f"dashboard missing BUY CSS class, got: {dashboard!r}"
    )
    assert "signal-sell" in dashboard, (
        f"dashboard missing SELL CSS class, got: {dashboard!r}"
    )

    # Both collected signals should end up in the session-state list in
    # the order the tickers were submitted.
    signals_state = final[_IDX_SIGNALS_STATE]
    assert signals_state == [buy_signal, sell_signal]


# ---------------------------------------------------------------------------
# Test 2: Partial failure renders both a signal card and an error card
# ---------------------------------------------------------------------------


def test_partial_failure_renders_error_card(buy_signal: FakeSignal) -> None:
    """One success + one failure renders both a signal card and an error card.

    The pipeline must not short-circuit on a single ticker failure:
    successful tickers keep their signal cards, and failed tickers
    render as error cards in the same dashboard. The combined layout
    lets the user see which tickers produced signals and which need to
    be retried.

    Validates: Requirements 5.1 (success card rendered) and 5.5 (error
    card rendered for failed ticker).
    """
    signals = {
        "AAPL": FakeCrewResult(success=True, signal=buy_signal),
        "FAIL": FakeCrewResult(
            success=False,
            signal=None,
            error="Connection error",
        ),
    }
    fake_crew, fake_callbacks = make_fake_crew_package(signals)

    results = _drive_run_analysis("AAPL, FAIL", fake_crew, fake_callbacks)

    assert len(results) >= 2, (
        f"Expected at least 2 yields (initial + ticker updates), "
        f"got {len(results)}"
    )

    final = results[-1]
    assert final[_IDX_ANALYZE_BTN].get("interactive") is True, (
        f"final yield must re-enable the button, got: {final[_IDX_ANALYZE_BTN]!r}"
    )

    dashboard = final[_IDX_SIGNALS_DASHBOARD]
    assert isinstance(dashboard, str)

    # Successful ticker: BUY card with green color class.
    assert "signal-buy" in dashboard, (
        f"dashboard missing BUY card for successful ticker, got: {dashboard!r}"
    )
    assert "AAPL" in dashboard

    # Failed ticker: dedicated error card with the provided message.
    assert "signal-error" in dashboard, (
        f"dashboard missing error CSS class for failed ticker, got: {dashboard!r}"
    )
    assert "FAIL" in dashboard, (
        f"dashboard missing failed-ticker symbol, got: {dashboard!r}"
    )
    assert "Connection error" in dashboard, (
        f"dashboard missing failure message, got: {dashboard!r}"
    )


# ---------------------------------------------------------------------------
# Test 3: Callback events translate into activity feed HTML entries
# ---------------------------------------------------------------------------


def test_activity_feed_receives_callback_events(
    buy_signal: FakeSignal,
    sample_activity_events: List[FakeActivityEvent],
) -> None:
    """``ActivityFeedCallback`` events are translated to activity feed HTML.

    The fake runner replays ``sample_activity_events`` through the
    callback handler before returning its per-ticker result. The
    handler's post-ticker drain should convert each buffered event into
    a ``render_activity_entry`` HTML fragment, and the ``task_start``
    event in the canned sequence must surface the ``⟳`` spinner glyph
    (Requirement 3.2).

    Validates: Requirements 3.1 (agent activity entries rendered), 3.3
    (intermediate outputs appended).
    """
    signals = {
        "AAPL": FakeCrewResult(success=True, signal=buy_signal),
    }

    # Always replay the canned sequence regardless of which ticker is
    # requested — the test only submits one ticker anyway, but this
    # keeps the fixture simple and reusable.
    fake_crew, fake_callbacks = make_fake_crew_package(
        signals,
        events_for_ticker=lambda ticker: sample_activity_events,
    )

    results = _drive_run_analysis("AAPL", fake_crew, fake_callbacks)

    final = results[-1]
    feed_html = final[_IDX_ACTIVITY_FEED]
    assert isinstance(feed_html, str)

    # Every canned event's agent name and message must show up in the
    # final feed HTML. This is a stronger check than "some entries
    # rendered" — it confirms the per-event translation loop fires for
    # every buffered event.
    for event in sample_activity_events:
        assert event.agent_name in feed_html, (
            f"activity feed missing agent name {event.agent_name!r} "
            f"for event {event!r}; feed: {feed_html!r}"
        )
        assert event.message in feed_html, (
            f"activity feed missing message {event.message!r} "
            f"for event {event!r}; feed: {feed_html!r}"
        )

    # The canned sequence includes exactly one ``task_start`` event, so
    # the spinner glyph must appear at least once in the rendered feed.
    assert "⟳" in feed_html, (
        f"activity feed missing spinner glyph for task_start event; "
        f"feed: {feed_html!r}"
    )
