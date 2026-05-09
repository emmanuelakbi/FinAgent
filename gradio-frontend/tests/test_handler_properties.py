"""Property-based tests for the ``run_analysis`` event handler in ``app.py``.

Each property mirrors a numbered correctness property from the
gradio-frontend design document and is linked to the originating
acceptance criterion via a ``Validates`` tag in the docstring.
"""

import string
import sys
import types
from unittest.mock import patch

import pytest
from hypothesis import given, settings, strategies as st

# ``app.py`` imports ``gradio`` at module import time. Skip this test file
# cleanly in environments that don't have gradio installed (e.g. when the
# other, gradio-free, property test files are being run in isolation).
pytest.importorskip("gradio")

import app as app_module  # noqa: E402  (import deferred until after importorskip)


# ---------------------------------------------------------------------------
# Shared helper strategies
# ---------------------------------------------------------------------------
#
# Exception messages are interpolated into an HTML error banner via an
# f-string, so to keep our substring assertion on the message reliable we
# restrict the message alphabet to printable ASCII characters that cannot
# be escaped or dropped by any downstream processing. We exclude:
#
#   * ``<``, ``>``, ``&`` — HTML-special characters that could be escaped
#     by a future hardening pass, which would break a naive ``in`` check,
#   * ``{`` and ``}`` — defensive, in case a future refactor threads the
#     message through ``str.format`` rather than an f-string.
#
# The alphabet still covers letters, digits, spaces, and most punctuation,
# which is more than enough to exercise the exception-handling path across
# the shape of messages produced by the real pipeline.

_SAFE_MESSAGE_CHARS = "".join(
    ch
    for ch in (string.ascii_letters + string.digits + string.punctuation + " ")
    if ch not in {"<", ">", "&", "{", "}"}
)


def _safe_exception_message() -> st.SearchStrategy[str]:
    """Non-empty exception message drawn from the HTML-safe alphabet."""
    return st.text(
        alphabet=_SAFE_MESSAGE_CHARS, min_size=1, max_size=40
    ).filter(lambda s: s.strip() != "")


# Exception classes worth exercising. We include a mix of built-in types
# that commonly surface from an LLM/networking pipeline so the property
# is not accidentally narrowed to one failure mode. Any ``Exception``
# subclass would work; the list is deliberately short to keep the
# shrinker fast.
_EXCEPTION_CLASSES = (
    RuntimeError,
    ValueError,
    ConnectionError,
    TimeoutError,
    KeyError,
    OSError,
)


# ---------------------------------------------------------------------------
# Fake ``crew`` package
# ---------------------------------------------------------------------------
#
# ``run_analysis`` imports its orchestration dependencies lazily, inside
# the ``try`` block that wraps the pipeline loop::
#
#     from crew import LLMConfig, OrchestratorConfig, WatchlistRunner
#     from crew.callbacks import ActivityEvent, ActivityFeedCallback, EventType
#
# That deferred import means we can inject a minimal stand-in for the
# package via ``sys.modules`` *before* the handler reaches the ``try``
# block, without having to install the real ``crewai`` dependency for the
# property test. The stand-in is engineered so that the pipeline is
# guaranteed to raise an exception from exactly one of two injection
# sites — runner construction or ``_run_single`` — which together cover
# both branches of the try/except block tested by Property 10.


def _make_fake_crew_modules(
    raise_on: str, exception: BaseException
) -> tuple[types.ModuleType, types.ModuleType]:
    """Build minimal ``crew`` and ``crew.callbacks`` stand-ins.

    Args:
        raise_on: ``"construction"`` to raise inside
            ``WatchlistRunner.__init__``, or ``"run_single"`` to raise
            from ``WatchlistRunner._run_single`` on the first ticker.
        exception: The exception instance to raise at the chosen site.

    Returns:
        A pair ``(crew, crew.callbacks)`` ready to be inserted into
        ``sys.modules``.
    """
    crew = types.ModuleType("crew")
    callbacks = types.ModuleType("crew.callbacks")

    class _LLMConfig:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            pass

    class _OrchestratorConfig:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class _ActivityEvent:
        pass

    class _ActivityFeedCallback:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class _EventType:
        # ``run_analysis`` compares ``event.event_type == EventType.TASK_START``
        # when translating pending events; a unique sentinel string is
        # enough to make that comparison meaningful.
        TASK_START = "task_start"

    if raise_on == "construction":

        class _WatchlistRunner:
            def __init__(self, *args, **kwargs) -> None:
                raise exception

    elif raise_on == "run_single":

        class _WatchlistRunner:
            def __init__(self, *args, **kwargs) -> None:
                # No-op so we reach the per-ticker loop.
                pass

            def _run_single(self, ticker):  # noqa: D401
                raise exception

    else:  # pragma: no cover - guarded by the strategy
        raise AssertionError(f"Unexpected raise_on value: {raise_on!r}")

    crew.LLMConfig = _LLMConfig
    crew.OrchestratorConfig = _OrchestratorConfig
    crew.WatchlistRunner = _WatchlistRunner
    callbacks.ActivityEvent = _ActivityEvent
    callbacks.ActivityFeedCallback = _ActivityFeedCallback
    callbacks.EventType = _EventType

    return crew, callbacks


# ---------------------------------------------------------------------------
# Property 10: Exception handling yields re-enabled button and error display
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=50)
@given(
    exception_class=st.sampled_from(_EXCEPTION_CLASSES),
    exception_message=_safe_exception_message(),
    raise_on=st.sampled_from(["construction", "run_single"]),
)
def test_exception_handling_yields_re_enabled_button(
    exception_class, exception_message, raise_on
):
    """Property 10: Exception handling yields re-enabled button and error display.

    For any exception raised during pipeline execution — either while
    constructing the ``WatchlistRunner`` or while running a single
    ticker through ``_run_single`` — the ``run_analysis`` generator must
    yield a final state where:

      * the Analyze button is interactive again
        (``gr.update(interactive=True)``), and
      * an error message is visible to the user, either in the error
        display banner (``gr.update(visible=True)`` with the exception
        message in its ``value``) or in the rendered activity feed HTML.

    We inject a fake ``crew`` / ``crew.callbacks`` package into
    ``sys.modules`` before the handler's deferred import executes so the
    exception fires at a controlled site. We also patch
    ``app.VLLM_ENDPOINT_URL`` to a truthy value so validation passes and
    the handler proceeds to the try/except wrapped pipeline block —
    otherwise the early configuration-error return would pre-empt the
    path this property is testing.

    Validates: Requirements 8.1, 8.3
    """
    exception = exception_class(exception_message)
    fake_crew, fake_callbacks = _make_fake_crew_modules(raise_on, exception)

    with patch.object(app_module, "VLLM_ENDPOINT_URL", "http://test-endpoint"), \
         patch.dict(
             sys.modules,
             {"crew": fake_crew, "crew.callbacks": fake_callbacks},
         ):
        # A single valid ticker is enough to drive the handler past both
        # validation steps and into the pipeline block. Using a fixed
        # ticker keeps the generated counterexamples small.
        results = list(
            app_module.run_analysis(
                ticker_input="AAPL",
                risk_tolerance="Moderate",
                portfolio_value=10000,
                trading_style="Swing Trading",
                activity_log=[],
                signals_state=[],
            )
        )

    # The handler must yield at least once — the exception path always
    # produces at least the initial "running" yield plus the final
    # exception-handler yield (two for construction failures, more when
    # the failure is in ``_run_single``).
    assert len(results) >= 1, "run_analysis should yield at least one update"

    final = results[-1]

    # The yield tuple layout is stable (see ``create_app``'s ``outputs=``)::
    #   (analyze_btn, error_display, progress_text,
    #    activity_feed, signals_dashboard,
    #    activity_log, signals_state)
    assert len(final) == 7, (
        f"Expected 7-tuple yield, got {len(final)}-tuple: {final!r}"
    )
    (
        analyze_btn_update,
        error_display_update,
        _progress_text_update,
        activity_feed_html,
        _signals_dashboard_html,
        _activity_log,
        _signals_state,
    ) = final

    # --- Analyze button is re-enabled (Requirement 8.1) --------------------
    # ``gr.update`` returns a plain dict in gradio 4.44.x, so ``interactive``
    # is a direct key lookup.
    assert isinstance(analyze_btn_update, dict), (
        f"Expected analyze button update to be a dict, got: "
        f"{type(analyze_btn_update).__name__}"
    )
    assert analyze_btn_update.get("interactive") is True, (
        f"Expected Analyze button to be re-enabled (interactive=True) in "
        f"the final yield, got: {analyze_btn_update!r}"
    )

    # --- Error is visible to the user (Requirements 8.1, 8.3) --------------
    # The design allows the error to surface in either the error-display
    # banner or the activity feed. We verify both claims:
    #
    #   1. At least one surface is actually visible (not hidden/no-op),
    #   2. The exception message appears verbatim on at least one
    #      surface so the user can read it.
    assert isinstance(error_display_update, dict), (
        f"Expected error display update to be a dict, got: "
        f"{type(error_display_update).__name__}"
    )
    error_banner_visible = error_display_update.get("visible") is True
    error_banner_value = str(error_display_update.get("value", ""))
    feed_html = activity_feed_html if isinstance(activity_feed_html, str) else ""

    assert error_banner_visible, (
        f"Expected error display to be visible in the final yield after an "
        f"exception, got error_display_update={error_display_update!r}"
    )

    # What the user actually sees on the banner / in the feed is
    # ``str(e)``, not the raw message passed to the exception constructor.
    # Some builtins (notably ``KeyError``) apply ``repr()`` to their
    # args, so ``KeyError('0\\0').__str__()`` renders as ``"'0\\\\0'"``
    # rather than the bare ``'0\\0'``. Compare against ``str(e)`` so the
    # property stays aligned with what a user actually reads.
    rendered_message = str(exception)

    message_on_banner = rendered_message in error_banner_value
    message_in_feed = rendered_message in feed_html
    assert message_on_banner or message_in_feed, (
        f"Expected the stringified exception {rendered_message!r} to appear in "
        f"either the error banner value or the activity feed HTML, got "
        f"error_banner_value={error_banner_value!r}, "
        f"activity_feed_html={feed_html!r}"
    )


# ---------------------------------------------------------------------------
# Property 6: Progress message contains ticker name and position
# ---------------------------------------------------------------------------

from app import _format_progress_message  # noqa: E402


# Tickers are drawn from uppercase letters and digits so the ticker
# cannot spuriously match any of the lowercase template tokens
# ("Analyzing", "ticker", "of", "elapsed", "s") used by
# ``_format_progress_message``.
_TICKER_ALPHABET_P6 = string.ascii_uppercase + string.digits


def _ticker_strategy_p6() -> st.SearchStrategy[str]:
    """Generate a non-empty uppercase-alphanumeric ticker symbol."""
    return st.text(alphabet=_TICKER_ALPHABET_P6, min_size=1, max_size=10)


@st.composite
def _position_and_total(draw) -> tuple[int, int]:
    """Draw ``(i, total)`` satisfying ``1 <= i <= total <= 50``.

    Producing the pair in a composite strategy keeps the dependency
    between ``i`` and ``total`` explicit so Hypothesis shrinks them
    jointly without ever reporting an invalid ``i > total`` combination.
    """
    total = draw(st.integers(min_value=1, max_value=50))
    i = draw(st.integers(min_value=1, max_value=total))
    return i, total


@given(
    ticker=_ticker_strategy_p6(),
    position=_position_and_total(),
    elapsed_seconds=st.integers(min_value=0, max_value=300),
)
def test_progress_message_contains_ticker_and_position(
    ticker, position, elapsed_seconds
):
    """Property 6: Progress message contains ticker name and position.

    For any ticker string, 1-indexed position ``i``, and total count
    ``M`` with ``1 <= i <= M``, the progress message produced by
    :func:`app._format_progress_message` must contain:

      * the ticker symbol verbatim (Requirement 4.2),
      * the string representation of ``i`` (Requirement 4.3),
      * the string representation of ``M`` (Requirement 4.3).

    Validates: Requirements 4.2, 4.3
    """
    i, total = position

    msg = _format_progress_message(
        ticker=ticker,
        i=i,
        total=total,
        elapsed_seconds=elapsed_seconds,
    )

    assert isinstance(msg, str)
    assert msg != ""

    assert ticker in msg, (
        f"Expected ticker {ticker!r} to appear in progress message, "
        f"got: {msg!r}"
    )
    assert str(i) in msg, (
        f"Expected position {i!r} to appear in progress message, "
        f"got: {msg!r}"
    )
    assert str(total) in msg, (
        f"Expected total {total!r} to appear in progress message, "
        f"got: {msg!r}"
    )
