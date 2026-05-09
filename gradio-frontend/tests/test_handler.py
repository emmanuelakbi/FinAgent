"""Unit tests for the ``run_analysis`` event handler in ``app.py``.

These tests exercise specific, named scenarios of the Analyze button
generator — configuration failure, ticker validation failure, portfolio
validation failure, pipeline timeout, and the "always re-enable the
button" invariant — by driving ``run_analysis`` directly and inspecting
its yields.

All paths tested here are short-circuit / failure paths: the handler
yields a single terminal tuple and returns. The happy-path streaming
behaviour is covered by integration tests (task 11.1) and
``test_handler_properties.py``.

Conventions used throughout this file:

* ``app.VLLM_ENDPOINT_URL`` is patched rather than ``os.environ`` because
  the module snapshots the value at import time; patching the env var
  after import has no effect.
* Yields are 7-tuples in the order wired up by ``analyze_btn.click``'s
  ``outputs=``: ``(analyze_btn, error_display, progress_text,
  activity_feed, signals_dashboard, activity_log, signals_state)``.
* ``gr.update(...)`` returns a plain ``dict`` in gradio 4.44.x, so
  fields are asserted via ``dict.get(...)``.
"""

import sys
import types
from unittest.mock import patch

import pytest

# ``app.py`` imports ``gradio`` at module import time. Skip this test
# file cleanly in environments where gradio isn't installed.
pytest.importorskip("gradio")

import app as app_module  # noqa: E402  (import deferred until after importorskip)


# ---------------------------------------------------------------------------
# Yield-tuple layout
# ---------------------------------------------------------------------------
#
# The handler's yield contract (see ``create_app``'s ``outputs=``):
#
#     (analyze_btn, error_display, progress_text,
#      activity_feed, signals_dashboard,
#      activity_log, signals_state)
#
# Indexed here as constants so the assertions below read cleanly.

_YIELD_TUPLE_LEN = 7
_IDX_ANALYZE_BTN = 0
_IDX_ERROR_DISPLAY = 1
_IDX_PROGRESS_TEXT = 2
_IDX_ACTIVITY_FEED = 3
_IDX_SIGNALS_DASHBOARD = 4
_IDX_ACTIVITY_LOG = 5
_IDX_SIGNALS_STATE = 6


# ---------------------------------------------------------------------------
# Fake ``crew`` package helpers
# ---------------------------------------------------------------------------
#
# ``run_analysis`` imports from ``crew`` lazily inside its try block, so
# we can inject a no-op stand-in via ``sys.modules`` to drive the
# handler past the deferred import without needing the real dependency
# installed. This is only required for the timeout test — every other
# test short-circuits before reaching the import.


def _make_noop_crew_modules() -> tuple[types.ModuleType, types.ModuleType]:
    """Build minimal ``crew`` / ``crew.callbacks`` stand-ins that don't raise.

    Used by the timeout test to drive ``run_analysis`` past its deferred
    ``from crew import ...`` block and into the per-ticker loop, where
    the ``TIMEOUT_SECONDS`` check lives. The returned ``WatchlistRunner``
    is safe to construct but will never have ``_run_single`` called on
    it — the timeout branch returns before that point.
    """
    crew = types.ModuleType("crew")
    callbacks = types.ModuleType("crew.callbacks")

    class _LLMConfig:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            pass

    class _OrchestratorConfig:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            pass

    class _ActivityEvent:  # noqa: D401 - unused marker class
        pass

    class _ActivityFeedCallback:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            pass

    class _EventType:
        # ``run_analysis`` only references ``EventType.TASK_START`` when
        # translating pending events, which never happens on the timeout
        # path, but the attribute must exist for the ``from`` import to
        # succeed.
        TASK_START = "task_start"

    class _WatchlistRunner:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            pass

        def _run_single(self, ticker):  # pragma: no cover - never reached
            raise AssertionError(
                "_run_single should not be called when the timeout check fires "
                "before the first ticker is dispatched"
            )

    crew.LLMConfig = _LLMConfig
    crew.OrchestratorConfig = _OrchestratorConfig
    crew.WatchlistRunner = _WatchlistRunner
    callbacks.ActivityEvent = _ActivityEvent
    callbacks.ActivityFeedCallback = _ActivityFeedCallback
    callbacks.EventType = _EventType

    return crew, callbacks


def _run(**overrides):
    """Invoke ``run_analysis`` with sensible defaults and collect yields.

    Defaults mirror a typical valid submission so each test only has to
    override the field it's exercising. Returns the list of yielded
    tuples.
    """
    kwargs = {
        "ticker_input": "AAPL",
        "risk_tolerance": "Moderate",
        "portfolio_value": 10000,
        "trading_style": "Swing Trading",
        "activity_log": [],
        "signals_state": [],
    }
    kwargs.update(overrides)
    return list(app_module.run_analysis(**kwargs))


# ---------------------------------------------------------------------------
# Configuration error: VLLM_ENDPOINT_URL not set
# ---------------------------------------------------------------------------


def test_vllm_endpoint_not_set_yields_config_error():
    """A missing ``VLLM_ENDPOINT_URL`` yields a single config-error update.

    Validates: Requirement 7.4 (missing endpoint surfaces a configuration
    error) and Requirement 8.1 (failure paths re-enable the Analyze
    button).
    """
    with patch.object(app_module, "VLLM_ENDPOINT_URL", None):
        results = _run()

    # The configuration-error branch short-circuits immediately, so
    # exactly one yield is expected.
    assert len(results) == 1, (
        f"Expected exactly one yield for the missing-endpoint path, "
        f"got {len(results)}: {results!r}"
    )

    final = results[-1]
    assert len(final) == _YIELD_TUPLE_LEN

    analyze_btn = final[_IDX_ANALYZE_BTN]
    error_display = final[_IDX_ERROR_DISPLAY]

    assert isinstance(analyze_btn, dict)
    assert analyze_btn.get("interactive") is True, (
        f"Analyze button must be re-enabled on config error, got: {analyze_btn!r}"
    )

    assert isinstance(error_display, dict)
    assert error_display.get("visible") is True, (
        f"Error display must be visible on config error, got: {error_display!r}"
    )
    assert "VLLM_ENDPOINT_URL" in str(error_display.get("value", "")), (
        f"Error message must mention VLLM_ENDPOINT_URL, got: {error_display!r}"
    )


def test_vllm_endpoint_empty_string_yields_config_error():
    """An empty-string ``VLLM_ENDPOINT_URL`` is treated the same as unset.

    Requirement 7.2 says the handler reads the variable via
    ``os.environ.get``, which returns ``""`` when a user sets the env
    var but leaves it blank. The handler uses a falsy check so both
    cases must surface the same config error.
    """
    with patch.object(app_module, "VLLM_ENDPOINT_URL", ""):
        results = _run()

    assert len(results) == 1
    final = results[-1]

    assert final[_IDX_ANALYZE_BTN].get("interactive") is True
    assert final[_IDX_ERROR_DISPLAY].get("visible") is True
    assert "VLLM_ENDPOINT_URL" in str(final[_IDX_ERROR_DISPLAY].get("value", ""))


# ---------------------------------------------------------------------------
# Ticker validation failures
# ---------------------------------------------------------------------------


def test_invalid_ticker_yields_validation_error():
    """Disallowed ticker characters yield a single validation-error update.

    Validates: Requirement 8.1 (button re-enabled on validation failure).
    The disallowed-character set is defined in ``validation.py`` as
    anything outside ``[A-Za-z0-9\\-\\.]``.
    """
    with patch.object(app_module, "VLLM_ENDPOINT_URL", "http://test-endpoint"):
        results = _run(ticker_input="$$$")

    assert len(results) == 1
    final = results[-1]

    analyze_btn = final[_IDX_ANALYZE_BTN]
    error_display = final[_IDX_ERROR_DISPLAY]

    assert analyze_btn.get("interactive") is True
    assert error_display.get("visible") is True

    error_value = str(error_display.get("value", ""))
    # The validation module emits messages of the form
    # "Invalid characters in: $$$ (invalid: $)" — assert the stable
    # "Invalid" prefix plus an echo of the offending character.
    assert "Invalid" in error_value, (
        f"Expected 'Invalid' in validation error, got: {error_value!r}"
    )
    assert "$" in error_value, (
        f"Expected offending character '$' in validation error, "
        f"got: {error_value!r}"
    )

    # The progress indicator must remain hidden on the validation path —
    # there's no analysis to show progress for.
    assert final[_IDX_PROGRESS_TEXT].get("visible") is False


def test_empty_ticker_yields_validation_error():
    """Empty ticker input yields a validation error prompting for at least one ticker.

    Validates: Requirements 1.2, 8.1.
    """
    with patch.object(app_module, "VLLM_ENDPOINT_URL", "http://test-endpoint"):
        results = _run(ticker_input="")

    assert len(results) == 1
    final = results[-1]

    assert final[_IDX_ANALYZE_BTN].get("interactive") is True
    error_display = final[_IDX_ERROR_DISPLAY]
    assert error_display.get("visible") is True
    # The validation module emits "Please enter at least one ticker symbol."
    # We assert on the stable substring "ticker" rather than the full
    # sentence so minor copy tweaks don't break the test.
    assert "ticker" in str(error_display.get("value", "")).lower()


def test_whitespace_only_ticker_yields_validation_error():
    """Whitespace-only ticker input is rejected the same as empty input."""
    with patch.object(app_module, "VLLM_ENDPOINT_URL", "http://test-endpoint"):
        results = _run(ticker_input="   \t  ")

    assert len(results) == 1
    final = results[-1]
    assert final[_IDX_ANALYZE_BTN].get("interactive") is True
    assert final[_IDX_ERROR_DISPLAY].get("visible") is True


# ---------------------------------------------------------------------------
# Portfolio value validation
# ---------------------------------------------------------------------------


def test_negative_portfolio_yields_error():
    """A negative portfolio value yields a validation-error update.

    Validates: Requirement 8.1. Portfolio validation runs after ticker
    validation, so we pair it with a known-valid ticker to ensure the
    portfolio branch is the one that fires.
    """
    with patch.object(app_module, "VLLM_ENDPOINT_URL", "http://test-endpoint"):
        results = _run(ticker_input="AAPL", portfolio_value=-100)

    assert len(results) == 1
    final = results[-1]

    assert final[_IDX_ANALYZE_BTN].get("interactive") is True

    error_display = final[_IDX_ERROR_DISPLAY]
    assert error_display.get("visible") is True
    # The validation module emits "Portfolio value must be non-negative."
    error_value = str(error_display.get("value", ""))
    assert "ortfolio" in error_value or "non-negative" in error_value, (
        f"Expected portfolio-value error, got: {error_value!r}"
    )


# ---------------------------------------------------------------------------
# Pipeline timeout
# ---------------------------------------------------------------------------


def test_timeout_yields_warning():
    """An elapsed time exceeding ``TIMEOUT_SECONDS`` yields a timeout warning.

    We force the timeout branch by patching ``TIMEOUT_SECONDS`` to ``-1``
    so the first ``elapsed = time.time() - analysis_start`` comparison
    (``elapsed > -1``) is trivially true before any ticker is
    dispatched. A no-op ``crew`` package is injected so the deferred
    import inside the handler's try block succeeds.

    Validates: Requirements 4.4 (timeout warning) and 8.1 (button
    re-enabled on timeout).
    """
    fake_crew, fake_callbacks = _make_noop_crew_modules()

    with patch.object(app_module, "VLLM_ENDPOINT_URL", "http://test-endpoint"), \
         patch.object(app_module, "TIMEOUT_SECONDS", -1), \
         patch.dict(
             sys.modules,
             {"crew": fake_crew, "crew.callbacks": fake_callbacks},
         ):
        results = _run(ticker_input="AAPL")

    # Expected sequence: initial "analysis started" yield (button
    # disabled, progress visible) followed by the timeout-branch yield
    # (button re-enabled, timeout banner visible).
    assert len(results) >= 2, (
        f"Expected at least two yields on the timeout path, got {len(results)}"
    )

    final = results[-1]
    assert len(final) == _YIELD_TUPLE_LEN

    # Button must be re-enabled so the user can retry with fewer tickers.
    assert final[_IDX_ANALYZE_BTN].get("interactive") is True

    error_display = final[_IDX_ERROR_DISPLAY]
    assert error_display.get("visible") is True
    error_value = str(error_display.get("value", ""))
    assert "timed out" in error_value.lower(), (
        f"Expected 'timed out' in timeout error banner, got: {error_value!r}"
    )

    # Progress indicator is hidden once the run terminates.
    assert final[_IDX_PROGRESS_TEXT].get("visible") is False

    # The activity feed should include a system timeout entry so the
    # user has an in-context record of why the run stopped.
    feed_html = final[_IDX_ACTIVITY_FEED]
    assert isinstance(feed_html, str)
    assert "timed out" in feed_html.lower(), (
        f"Expected timeout entry in activity feed HTML, got: {feed_html!r}"
    )


# ---------------------------------------------------------------------------
# Invariant: final yield always re-enables the button
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario_id, vllm_url, ticker_input, portfolio_value",
    [
        ("missing_endpoint", None, "AAPL", 10000),
        ("empty_endpoint", "", "AAPL", 10000),
        ("invalid_ticker", "http://test-endpoint", "$$$", 10000),
        ("empty_ticker", "http://test-endpoint", "", 10000),
        ("negative_portfolio", "http://test-endpoint", "AAPL", -100),
    ],
    ids=lambda v: v if isinstance(v, str) else repr(v),
)
def test_final_yield_always_has_reenabled_button(
    scenario_id, vllm_url, ticker_input, portfolio_value
):
    """Every failure path must leave the Analyze button re-enabled.

    This enforces the invariant described in Requirement 8.4: regardless
    of which validation or configuration check fails, the user must be
    able to correct their input and retry without reloading the page.

    The timeout branch is covered separately by
    :func:`test_timeout_yields_warning` because it requires the fake
    crew injection and isn't expressible as a pure parameter tuple.
    """
    with patch.object(app_module, "VLLM_ENDPOINT_URL", vllm_url):
        results = _run(ticker_input=ticker_input, portfolio_value=portfolio_value)

    assert len(results) >= 1, f"[{scenario_id}] handler yielded nothing"

    final = results[-1]
    assert len(final) == _YIELD_TUPLE_LEN, (
        f"[{scenario_id}] expected {_YIELD_TUPLE_LEN}-tuple, got: {final!r}"
    )

    analyze_btn = final[_IDX_ANALYZE_BTN]
    assert isinstance(analyze_btn, dict), (
        f"[{scenario_id}] expected dict update, got: {type(analyze_btn).__name__}"
    )
    assert analyze_btn.get("interactive") is True, (
        f"[{scenario_id}] Analyze button must be re-enabled in final yield, "
        f"got: {analyze_btn!r}"
    )

    # Every failure scenario should also surface a visible error banner
    # so the user understands why the run didn't proceed.
    error_display = final[_IDX_ERROR_DISPLAY]
    assert isinstance(error_display, dict)
    assert error_display.get("visible") is True, (
        f"[{scenario_id}] error display must be visible in final yield, "
        f"got: {error_display!r}"
    )
