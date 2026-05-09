"""FinAgent Gradio frontend — application entry point.

This module constructs the Gradio Blocks application used as the FinAgent
Hugging Face Space. The outer shell (title, theme, custom CSS) is defined
in :func:`create_app`; UI widgets, session state, and event wiring are
added by later tasks (6.2, 6.3, 6.4).

The vLLM inference endpoint URL is read from the ``VLLM_ENDPOINT_URL``
environment variable at module import time so it can be validated early
by the Analyze event handler.
"""

import os
import time
from datetime import datetime

import gradio as gr

from rendering import (
    build_css,
    render_activity_entry,
    render_activity_feed,
    render_error_card,
    render_signal_card,
    render_summary,
)
from validation import validate_portfolio_value, validate_tickers

# Inference endpoint URL for the vLLM server powering the agent pipeline.
# Read at import time; the event handler is responsible for surfacing a
# user-facing error when this value is missing.
VLLM_ENDPOINT_URL = os.environ.get("VLLM_ENDPOINT_URL")

# Hard cap on total pipeline execution time per Requirement 4.4. The
# pipeline loop checks the elapsed wall-clock time before dispatching
# each ticker and yields a timeout warning once this budget is exceeded.
TIMEOUT_SECONDS = 180


def _format_progress_message(
    ticker: str,
    i: int,
    total: int,
    elapsed_seconds: int,
) -> str:
    """Format the in-flight progress message shown above the activity feed.

    Pulled out of :func:`run_analysis` as a pure, side-effect-free helper
    so the progress string's contract (Requirements 4.2, 4.3) can be
    exercised directly by property tests without constructing a full
    Gradio event loop.

    Args:
        ticker: The ticker symbol currently being analyzed.
        i: 1-indexed position of ``ticker`` in the watchlist.
        total: Total number of tickers in the watchlist (``i <= total``).
        elapsed_seconds: Whole seconds elapsed since the analysis began;
            rendered verbatim in the ``(elapsed: Ns)`` suffix.

    Returns:
        A Markdown-formatted string of the form::

            **Analyzing ticker {i} of {total}** — {ticker} (elapsed: {N}s)

        which contains the ticker, ``i``, and ``total`` as substrings
        (Property 6 in the design document).
    """
    return (
        f"**Analyzing ticker {i} of {total}** — {ticker} "
        f"(elapsed: {elapsed_seconds}s)"
    )


def run_analysis(
    ticker_input,
    risk_tolerance,
    portfolio_value,
    trading_style,
    activity_log,
    signals_state,
):
    """Analyze button event handler — streams UI updates as a generator.

    This is the task 7.1 slice of the full handler: it covers environment
    and input validation plus initial state setup. The pipeline execution
    loop (task 7.2), error handling / completion yields (task 7.3), and
    the ``_render_signals_dashboard`` helper (task 7.4) are added by
    subsequent tasks.

    Each ``yield`` produces the tuple wired to ``analyze_btn.click``'s
    outputs in :func:`create_app`, in this order::

        (analyze_btn, error_display, progress_text,
         activity_feed, signals_dashboard,
         activity_log, signals_state)

    Yields:
        Tuples of :class:`gradio.update` calls and session-state values
        that incrementally update the UI during analysis.
    """
    # --- Environment validation ---
    # The vLLM endpoint URL is required to run the pipeline. Surface a
    # clear configuration error and re-enable the button so the user can
    # retry once the variable is set.
    if not VLLM_ENDPOINT_URL:
        yield (
            gr.update(interactive=True),
            gr.update(
                value=(
                    "❌ Configuration error: `VLLM_ENDPOINT_URL` "
                    "environment variable is not set."
                ),
                visible=True,
            ),
            gr.update(visible=False),
            gr.update(),
            gr.update(),
            activity_log,
            signals_state,
        )
        return

    # --- Ticker validation ---
    validation = validate_tickers(ticker_input)
    if not validation.valid:
        yield (
            gr.update(interactive=True),
            gr.update(value=f"❌ {validation.error_message}", visible=True),
            gr.update(visible=False),
            gr.update(),
            gr.update(),
            activity_log,
            signals_state,
        )
        return

    # --- Portfolio value validation ---
    portfolio_error = validate_portfolio_value(portfolio_value)
    if portfolio_error:
        yield (
            gr.update(interactive=True),
            gr.update(value=f"❌ {portfolio_error}", visible=True),
            gr.update(visible=False),
            gr.update(),
            gr.update(),
            activity_log,
            signals_state,
        )
        return

    # --- Initialize per-run state ---
    # Reset activity log and signals for a fresh run; capture the start
    # timestamp so the pipeline loop (task 7.2) can enforce the timeout
    # and report elapsed time in progress updates.
    activity_log = []
    signals_state = []
    analysis_start = time.time()
    tickers = validation.tickers
    total = len(tickers)

    # Seed the activity feed with a start entry so the user sees
    # immediate feedback the moment the button is clicked.
    start_entry = render_activity_entry(
        datetime.now(),
        "System",
        f"Analysis started for {total} ticker(s)",
        False,
    )
    activity_log.append(start_entry)

    # Initial yield: disable button to prevent re-submission, hide any
    # stale error banner, show the progress indicator, seed the activity
    # feed, and clear the signals dashboard from any previous run.
    yield (
        gr.update(interactive=False),
        gr.update(visible=False),
        gr.update(value=f"**Analyzing ticker 1 of {total}**", visible=True),
        render_activity_feed(activity_log),
        "",
        activity_log,
        signals_state,
    )

    # --- Pipeline execution loop (task 7.2) ---
    # The entire pipeline setup + per-ticker loop runs inside a single
    # ``try`` block (task 7.3) so any unexpected exception — import
    # failure, runner construction error, or an unhandled crash inside
    # ``_run_single`` — is surfaced to the user as a visible error and
    # the Analyze button is re-enabled instead of leaving the UI in a
    # locked, spinning state (Requirements 8.1, 8.3).
    try:
        # Import the orchestration package lazily so ``app.py`` remains
        # importable in environments where ``crew`` isn't installed
        # (e.g., isolated unit-test runs of the validation/rendering
        # modules).
        from crew import (
            LLMConfig,
            OrchestratorConfig,
            WatchlistRunner,
        )
        from crew.callbacks import ActivityEvent, ActivityFeedCallback, EventType

        # Configure the orchestrator to point at the vLLM inference endpoint.
        config = OrchestratorConfig(
            llm=LLMConfig(base_url=VLLM_ENDPOINT_URL),
        )

        # Buffer events emitted by the runner/crew during ``_run_single``.
        # The callback handler runs synchronously on the same thread, so
        # a plain list plus a closure is sufficient — we drain it into
        # the activity feed after each ticker completes.
        pending_events: list[ActivityEvent] = []

        def event_handler(event: ActivityEvent) -> None:
            pending_events.append(event)

        callback = ActivityFeedCallback(handler=event_handler)
        runner = WatchlistRunner(config=config, tools={}, callback=callback)

        for i, ticker in enumerate(tickers, 1):
            # Enforce the overall pipeline timeout (Requirement 4.4). We
            # check before dispatching each ticker so a slow ticker
            # can't push the total past the budget unnoticed; when
            # exceeded we append a timeout entry, re-enable the Analyze
            # button, and return early.
            elapsed = time.time() - analysis_start
            if elapsed > TIMEOUT_SECONDS:
                timeout_entry = render_activity_entry(
                    datetime.now(),
                    "System",
                    f"⚠️ Analysis timed out after {TIMEOUT_SECONDS}s",
                    False,
                )
                activity_log.append(timeout_entry)
                yield (
                    gr.update(interactive=True),
                    gr.update(
                        value="⚠️ Analysis timed out. Try fewer tickers.",
                        visible=True,
                    ),
                    gr.update(visible=False),
                    render_activity_feed(activity_log),
                    _render_signals_dashboard(signals_state),
                    activity_log,
                    signals_state,
                )
                return

            # Progress message surfaces the current ticker, its position
            # in the batch, and elapsed seconds (Requirements 4.2, 4.3,
            # 3.6). The format is centralized in
            # :func:`_format_progress_message` so the contract can be
            # property-tested directly.
            progress_msg = _format_progress_message(
                ticker=ticker,
                i=i,
                total=total,
                elapsed_seconds=int(elapsed),
            )

            # Run the crew pipeline for this ticker. ``_run_single``
            # handles its own exception isolation and always returns a
            # CrewResult (success or failure), so per-ticker errors
            # won't break the outer loop.
            result = runner._run_single(ticker)

            # Translate buffered callback events into activity feed HTML
            # entries. Task-start events show a spinner to indicate work
            # in progress; all other event types render as static lines.
            for event in pending_events:
                entry = render_activity_entry(
                    event.timestamp,
                    event.agent_name,
                    event.message,
                    is_spinner=(event.event_type == EventType.TASK_START),
                )
                activity_log.append(entry)
            pending_events.clear()

            # Collect the per-ticker outcome. Successful runs contribute
            # a TradingSignal; failures contribute an error dict that
            # ``_render_signals_dashboard`` renders via
            # ``render_error_card`` (Requirement 5.5).
            if result.success and result.signal is not None:
                signals_state.append(result.signal)
            else:
                signals_state.append(
                    {
                        "ticker": ticker,
                        "error": result.error or "Unknown error",
                    }
                )

            # Stream the intermediate state to the UI: keep the button
            # disabled (still running), hide the error banner, refresh
            # the progress line, activity feed, and signals dashboard.
            yield (
                gr.update(interactive=False),
                gr.update(visible=False),
                gr.update(value=progress_msg, visible=True),
                render_activity_feed(activity_log),
                _render_signals_dashboard(signals_state),
                activity_log,
                signals_state,
            )

    except Exception as e:
        # --- Unhandled pipeline error (task 7.3) ---
        # Log the failure to the activity feed so the user can see what
        # went wrong in-context (Requirement 8.2), surface a visible
        # error banner with the exception message, hide the progress
        # indicator, and re-enable the Analyze button so the user can
        # retry (Requirements 8.1, 8.3). Any signals already collected
        # from earlier tickers are preserved in ``signals_state`` and
        # re-rendered in the final yield.
        error_entry = render_activity_entry(
            datetime.now(),
            "System",
            f"❌ Error: {str(e)}",
            False,
        )
        activity_log.append(error_entry)
        yield (
            gr.update(interactive=True),
            gr.update(value=f"❌ An error occurred: {str(e)}", visible=True),
            gr.update(visible=False),
            render_activity_feed(activity_log),
            _render_signals_dashboard(signals_state),
            activity_log,
            signals_state,
        )
        return

    # --- Successful completion (task 7.3) ---
    # The pipeline finished without raising. Append a completion entry
    # reporting total elapsed time (Requirement 8.4), hide the progress
    # indicator, clear any lingering error banner, and re-enable the
    # Analyze button for the next run.
    elapsed_total = time.time() - analysis_start
    complete_entry = render_activity_entry(
        datetime.now(),
        "System",
        f"✅ Analysis complete ({int(elapsed_total)}s)",
        False,
    )
    activity_log.append(complete_entry)

    yield (
        gr.update(interactive=True),
        gr.update(visible=False),
        gr.update(visible=False),
        render_activity_feed(activity_log),
        _render_signals_dashboard(signals_state),
        activity_log,
        signals_state,
    )


def _render_signals_dashboard(signals: list) -> str:
    """Render the full signals dashboard HTML (summary bar + cards).

    Iterates the collected results and produces a dashboard composed of
    an aggregate :func:`render_summary` bar followed by one HTML card per
    result: a :func:`render_signal_card` for :class:`TradingSignal`-like
    objects and a :func:`render_error_card` for error dicts of the form
    ``{"ticker": ..., "error": ...}``.

    The action label is read defensively via ``getattr(item.action,
    "value", item.action)`` so this helper works both with the orchestrator's
    ``Action`` enum (``Action.BUY.value == "BUY"``) and with plain string
    actions, without importing from the ``crew`` package.

    Args:
        signals: Ordered list of :class:`TradingSignal`-like objects and/or
            error dicts accumulated during an analysis run.

    Returns:
        Concatenated HTML string (summary bar followed by card HTML joined
        by newlines), or an empty string when ``signals`` is empty.
    """
    if not signals:
        return ""

    cards: list[str] = []
    buy_count = 0
    sell_count = 0
    hold_count = 0

    for item in signals:
        # Error dicts are produced when a ticker fails analysis — render
        # them as dedicated error cards and skip action counting.
        if isinstance(item, dict) and "error" in item:
            cards.append(render_error_card(item["ticker"], item["error"]))
            continue

        cards.append(render_signal_card(item))

        # Categorize the action for the summary bar. ``item.action`` may
        # be an enum (``.value`` gives the label) or already a string —
        # ``getattr`` with a default covers both without importing the
        # orchestrator's ``Action`` enum.
        action_label = str(getattr(item.action, "value", item.action)).upper()
        if action_label == "BUY":
            buy_count += 1
        elif action_label == "SELL":
            sell_count += 1
        elif action_label == "HOLD":
            hold_count += 1

    total = len(signals)
    summary = render_summary(total, buy_count, sell_count, hold_count)

    return summary + "\n".join(cards)


def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks application.

    Creates the outer shell: page title, dark Base theme with an emerald
    primary hue and slate neutral hue, and the custom dark financial
    terminal CSS from :func:`rendering.build_css`.

    Returns:
        The configured :class:`gradio.Blocks` instance, ready to have UI
        widgets and event handlers attached by subsequent tasks.
    """
    custom_css = build_css()

    with gr.Blocks(
        title="FinAgent - AI Trading Signals",
        theme=gr.themes.Base(
            primary_hue="emerald",
            neutral_hue="slate",
        ),
        css=custom_css,
    ) as app:
        # --- Header ---
        gr.Markdown(
            "# 🤖 FinAgent\n"
            "### AI-Powered Trading Signal Generator"
        )

        # --- Session State (per-user, isolated by gr.State) ---
        activity_log = gr.State([])      # list[str]: HTML strings for activity feed entries
        signals_state = gr.State([])     # list[TradingSignal | dict]: collected signals / errors
        start_time = gr.State(None)      # Optional[float]: time.time() when analysis started

        # --- Main Layout: Input panel (left) + Activity/Signals (right) ---
        with gr.Row():
            # Left Column: Input Panel
            with gr.Column(scale=1):
                ticker_input = gr.Textbox(
                    label="Watchlist",
                    placeholder="AAPL, NVDA, TSLA, BTC-USD",
                    info="Comma-separated tickers (max 10)",
                )
                risk_tolerance = gr.Dropdown(
                    choices=["Conservative", "Moderate", "Aggressive"],
                    value="Moderate",
                    label="Risk Tolerance",
                )
                portfolio_value = gr.Number(
                    value=10000,
                    minimum=0,
                    label="Portfolio Value ($)",
                )
                trading_style = gr.Dropdown(
                    choices=["Day Trading", "Swing Trading", "Position Trading"],
                    value="Swing Trading",
                    label="Trading Style",
                )
                analyze_btn = gr.Button(
                    "🔍 Analyze",
                    variant="primary",
                    interactive=True,
                )
                error_display = gr.Markdown(visible=False)

            # Right Column: Activity Feed + Signals Dashboard
            with gr.Column(scale=2):
                progress_text = gr.Markdown(visible=False)
                activity_feed = gr.HTML(
                    label="Agent Activity",
                    value="<div class='activity-feed'></div>",
                )
                signals_dashboard = gr.HTML(
                    label="Trading Signals",
                    value="",
                )

        # --- Event Wiring ---
        # The Analyze button streams results from the ``run_analysis``
        # generator. Inputs carry the user's configuration plus the
        # per-session activity log and collected signals so the handler
        # can extend them incrementally. Outputs cover every widget the
        # handler mutates during the run (button interactivity, error
        # banner, progress text, activity feed, signals dashboard) plus
        # the two session-state values it updates.
        analyze_btn.click(
            fn=run_analysis,
            inputs=[
                ticker_input,
                risk_tolerance,
                portfolio_value,
                trading_style,
                activity_log,
                signals_state,
            ],
            outputs=[
                analyze_btn,
                error_display,
                progress_text,
                activity_feed,
                signals_dashboard,
                activity_log,
                signals_state,
            ],
        )

        # --- Footer ---
        gr.Markdown(
            "⚠️ **Disclaimer:** Trading signals are for informational purposes only "
            "and do not constitute financial advice. Always do your own research."
        )

    # Expose widgets and session state on the returned Blocks so subsequent
    # tasks (6.3 event wiring) can reference them without re-entering the
    # context. These attributes are an internal contract between tasks in
    # this module only.
    app.ticker_input = ticker_input
    app.risk_tolerance = risk_tolerance
    app.portfolio_value = portfolio_value
    app.trading_style = trading_style
    app.analyze_btn = analyze_btn
    app.error_display = error_display
    app.progress_text = progress_text
    app.activity_feed = activity_feed
    app.signals_dashboard = signals_dashboard
    app.activity_log = activity_log
    app.signals_state = signals_state
    app.start_time = start_time

    return app


if __name__ == "__main__":
    # Launch the Gradio app on all interfaces so the Hugging Face Space
    # container can route external traffic to it (per Requirement 7.5).
    create_app().launch(server_name="0.0.0.0")
