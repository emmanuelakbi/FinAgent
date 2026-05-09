# Implementation Plan: Gradio Frontend

## Overview

This plan implements the FinAgent Gradio frontend — a dark-themed financial terminal dashboard deployed as a Hugging Face Space. The implementation proceeds incrementally: project structure and dependencies first, then pure validation/rendering modules (testable in isolation), followed by the Gradio layout, the generator-based event handler, deployment config, and finally tests. Each step builds on the previous and integrates immediately.

## Tasks

- [x] 1. Set up project structure and dependencies
  - [x] 1.1 Create `requirements.txt` with pinned dependencies
    - Create `gradio-frontend/requirements.txt` with: `gradio==4.44.1`, `crewai==0.80.0`, `langchain-openai==0.2.14`
    - Add `hypothesis` as a dev/test dependency
    - _Requirements: 7.3_

  - [x] 1.2 Create `gradio-frontend/validation.py` module skeleton
    - Create the file with imports (`re`, `dataclasses`, `typing`)
    - Define the `ValidationResult` dataclass with fields: `valid`, `tickers`, `error_message`
    - Define the `TICKER_PATTERN` regex constant and `MAX_TICKERS = 10`
    - Add function stubs for `validate_tickers` and `validate_portfolio_value`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.2_

  - [x] 1.3 Create `gradio-frontend/rendering.py` module skeleton
    - Create the file with imports (`datetime`, `typing`)
    - Add function stubs for `build_css`, `render_signal_card`, `render_error_card`, `render_summary`, `render_activity_entry`, `render_activity_feed`
    - _Requirements: 5.1, 5.2, 5.3, 5.5, 6.1, 6.4_

- [x] 2. Implement validation module
  - [x] 2.1 Implement `validate_tickers` function
    - Handle empty/whitespace-only input → return `valid=False` with error message
    - Split on commas, trim whitespace, convert to uppercase
    - Filter out empty segments after split
    - Validate each ticker against `TICKER_PATTERN` regex; collect invalid characters
    - Enforce `MAX_TICKERS` limit (10)
    - Return `ValidationResult` with normalized tickers on success
    - _Requirements: 1.2, 1.3, 1.4, 1.5_

  - [x] 2.2 Implement `validate_portfolio_value` function
    - Return error message string if value < 0
    - Return `None` if value is valid (non-negative)
    - _Requirements: 2.2_

  - [x] 2.3 Write property test: ticker normalization preserves content
    - **Property 1: Ticker normalization preserves content**
    - Use Hypothesis to generate comma-separated strings of valid ticker characters with arbitrary whitespace
    - Assert returned tickers are trimmed uppercase versions of input segments
    - Assert count matches non-empty segments
    - **Validates: Requirements 1.3**

  - [x] 2.4 Write property test: empty/whitespace input is rejected
    - **Property 2: Empty and whitespace-only input is rejected**
    - Use Hypothesis `text(alphabet=whitespace_chars)` strategy
    - Assert `valid=False`, non-empty `error_message`, empty `tickers` list
    - **Validates: Requirements 1.2**

  - [x] 2.5 Write property test: invalid characters are detected
    - **Property 3: Invalid characters are detected and reported**
    - Generate ticker strings containing at least one character outside `[A-Za-z0-9\-\.]`
    - Assert `valid=False` and error message contains at least one of the invalid characters
    - **Validates: Requirements 1.4**

  - [x] 2.6 Write property test: maximum ticker count enforced
    - **Property 4: Maximum ticker count is enforced**
    - Generate lists of more than 10 valid ticker symbols joined by commas
    - Assert `valid=False` and error message mentions the maximum limit
    - **Validates: Requirements 1.5**

- [x] 3. Checkpoint - Validation module complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement rendering module
  - [x] 4.1 Implement `build_css` function
    - Return the complete CSS string for dark financial terminal theme
    - Include styles for: `.gradio-container`, `.activity-feed`, `.activity-entry`, `.activity-timestamp`, `.activity-agent`, `.activity-spinner`, `.signal-card`, `.signal-buy/.signal-sell/.signal-hold/.signal-error`, `.signal-ticker`, `.signal-action-*`, `.signal-confidence`, `.signal-prices`, `.signal-price-item`, `.summary-bar`, `.summary-item`
    - Use monospace font family (`JetBrains Mono`, `Fira Code`, `Courier New`)
    - Use dark background colors (`#0d1117`, `#161b22`)
    - _Requirements: 6.1, 6.4_

  - [x] 4.2 Implement `render_signal_card` function
    - Accept a `TradingSignal` object
    - Render HTML card with: ticker name, action text, confidence percentage, entry/stop-loss/target prices grid, reasoning list
    - Apply CSS class `signal-buy` for BUY, `signal-sell` for SELL, `signal-hold` for HOLD
    - Apply action color classes for the action text
    - Handle optional price fields (show "N/A" if missing)
    - _Requirements: 5.1, 5.2, 5.4_

  - [x] 4.3 Implement `render_error_card` function
    - Accept ticker string and error message string
    - Render HTML card with `signal-error` class, ticker name, and error message
    - _Requirements: 5.5_

  - [x] 4.4 Implement `render_summary` function
    - Accept total, buy_count, sell_count, hold_count integers
    - Render summary bar with color-coded counts (green for BUY, red for SELL, yellow for HOLD)
    - _Requirements: 5.3_

  - [x] 4.5 Implement `render_activity_entry` function
    - Accept datetime timestamp, agent_name, message, and is_spinner boolean
    - Format timestamp as `HH:MM:SS`
    - Include spinner element (`⟳`) only when `is_spinner=True`
    - Wrap in `.activity-entry` div with timestamp, agent name, and message spans
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 4.6 Implement `render_activity_feed` function
    - Accept list of HTML entry strings
    - Wrap in `.activity-feed` container div
    - Include auto-scroll JavaScript snippet
    - _Requirements: 3.5_

  - [x] 4.7 Write property test: activity entry rendering
    - **Property 5: Activity entry rendering includes timestamp, agent name, and spinner control**
    - Generate arbitrary datetime, non-empty agent name, non-empty message, boolean spinner flag
    - Assert output contains formatted timestamp, agent name, message, and spinner iff `is_spinner=True`
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4**

  - [x] 4.8 Write property test: signal card renders all fields with correct color
    - **Property 7: Signal card renders all fields with correct action color class**
    - Generate valid TradingSignal with action in {BUY, SELL, HOLD}, confidence 0-100, non-negative prices
    - Assert HTML contains ticker, action text, confidence, prices, and correct CSS class
    - **Validates: Requirements 5.1, 5.2, 5.4**

  - [x] 4.9 Write property test: summary counts consistency
    - **Property 8: Summary counts are consistent with signal list**
    - Generate arbitrary counts where buy + sell + hold == total
    - Assert rendered HTML contains all count values
    - **Validates: Requirements 5.3**

  - [x] 4.10 Write property test: error card contains ticker and message
    - **Property 9: Error card contains ticker and error message**
    - Generate non-empty ticker and non-empty error message strings
    - Assert rendered HTML contains both the ticker and the error message
    - **Validates: Requirements 5.5**

- [x] 5. Checkpoint - Rendering module complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement Gradio app layout
  - [x] 6.1 Create `gradio-frontend/app.py` with `create_app` function
    - Import `os`, `gradio`, validation module, rendering module
    - Read `VLLM_ENDPOINT_URL` from environment
    - Create `gr.Blocks` with title "FinAgent - AI Trading Signals", dark Base theme (primary_hue="emerald", neutral_hue="slate"), and custom CSS from `build_css()`
    - _Requirements: 6.1, 6.2, 7.1_

  - [x] 6.2 Build the UI layout within `create_app`
    - Add header Markdown with FinAgent title and branding
    - Create session state: `activity_log = gr.State([])`, `signals_state = gr.State([])`, `start_time = gr.State(None)`
    - Left column (scale=1): ticker Textbox with placeholder, Risk Tolerance dropdown (Conservative/Moderate/Aggressive), Portfolio Value number input (default 10000, min 0), Trading Style dropdown (Day/Swing/Position Trading), Analyze button (primary variant), error display Markdown (hidden)
    - Right column (scale=2): progress text Markdown (hidden), activity feed HTML, signals dashboard HTML
    - Add disclaimer footer Markdown
    - _Requirements: 1.1, 2.1, 2.2, 2.3, 2.4, 6.2, 6.3_

  - [x] 6.3 Wire the Analyze button click event
    - Connect `analyze_btn.click` to `run_analysis` generator function
    - Set inputs: ticker_input, risk_tolerance, portfolio_value, trading_style, activity_log, signals_state
    - Set outputs: analyze_btn, error_display, progress_text, activity_feed, signals_dashboard, activity_log, signals_state
    - _Requirements: 2.4, 2.5_

  - [x] 6.4 Add application launch block
    - Add `if __name__ == "__main__"` block
    - Call `create_app().launch(server_name="0.0.0.0")`
    - _Requirements: 7.1, 7.5_

- [x] 7. Implement event handler (generator function)
  - [x] 7.1 Implement `run_analysis` generator — validation and initialization
    - Check `VLLM_ENDPOINT_URL` is set; yield config error if missing
    - Call `validate_tickers` and `validate_portfolio_value`; yield validation errors if invalid
    - Initialize `activity_log = []`, `signals_state = []`, record `analysis_start = time.time()`
    - Yield initial state: disable button, hide errors, show progress, clear signals
    - _Requirements: 7.2, 7.4, 8.1_

  - [x] 7.2 Implement `run_analysis` generator — pipeline execution loop
    - Configure `OrchestratorConfig` with `VLLM_ENDPOINT_URL`
    - Set up `ActivityFeedCallback` with event collection handler
    - Loop through validated tickers:
      - Check elapsed time against `TIMEOUT_SECONDS = 180`; yield timeout warning if exceeded
      - Update progress message with ticker name and position ("Analyzing ticker N of M")
      - Call `runner._run_single(ticker)` for each ticker
      - Process pending callback events into activity feed entries
      - Collect signal or error dict into `signals_state`
      - Yield intermediate update with current state
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.6, 4.1, 4.2, 4.3, 4.4_

  - [x] 7.3 Implement `run_analysis` generator — error handling and completion
    - Wrap pipeline loop in try/except; on exception: log error to activity feed, yield re-enabled button with error message
    - After successful loop: append completion entry to activity feed, yield final state with re-enabled button and hidden progress
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

  - [x] 7.4 Implement `_render_signals_dashboard` helper function
    - Iterate signals list; render signal cards for `TradingSignal` objects, error cards for error dicts
    - Count BUY/SELL/HOLD actions
    - Prepend summary bar via `render_summary`
    - Return concatenated HTML string
    - _Requirements: 5.1, 5.3, 5.5_

  - [x] 7.5 Write property test: progress message format
    - **Property 6: Progress message contains ticker name and position**
    - Generate ticker string, position i (1 ≤ i ≤ M), total M
    - Assert progress message contains ticker, i, and M
    - **Validates: Requirements 4.2, 4.3**

  - [x] 7.6 Write property test: exception handling yields re-enabled button
    - **Property 10: Exception handling yields re-enabled button and error display**
    - Mock pipeline to raise arbitrary exceptions
    - Assert final yield has interactive=True button and visible error message
    - **Validates: Requirements 8.1, 8.3**

- [x] 8. Checkpoint - Core implementation complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Hugging Face Space deployment configuration
  - [x] 9.1 Create `gradio-frontend/README.md` with HF Space metadata
    - Add YAML frontmatter with: `title: FinAgent`, `emoji: 🤖`, `sdk: gradio`, `sdk_version: 4.44.1`, `app_file: app.py`
    - Add brief description of the application
    - _Requirements: 7.1_

- [x] 10. Unit tests
  - [x] 10.1 Write unit tests for validation module (`tests/test_validation.py`)
    - Test specific examples: empty string, single valid ticker, multiple tickers with whitespace, invalid characters (e.g., `$`, `@`), exactly 10 tickers (valid), 11 tickers (invalid)
    - Test `validate_portfolio_value` with negative value, zero, positive value
    - _Requirements: 1.2, 1.3, 1.4, 1.5, 2.2_

  - [x] 10.2 Write unit tests for rendering module (`tests/test_rendering.py`)
    - Test `build_css` returns string containing dark background colors and monospace font
    - Test `render_signal_card` with a BUY signal — verify green class, all fields present
    - Test `render_signal_card` with SELL and HOLD signals — verify red/yellow classes
    - Test `render_error_card` — verify error class and message
    - Test `render_summary` — verify all counts displayed
    - Test `render_activity_feed` — verify scroll script included
    - _Requirements: 5.1, 5.2, 5.3, 5.5, 6.1, 6.4_

  - [x] 10.3 Write unit tests for event handler (`tests/test_handler.py`)
    - Test `VLLM_ENDPOINT_URL` not set → yields config error with button re-enabled
    - Test invalid ticker input → yields validation error with button re-enabled
    - Mock `time.time()` to trigger 180s timeout → verify timeout warning yielded
    - Test button state: always re-enabled in final yield regardless of outcome
    - _Requirements: 7.2, 7.4, 8.1, 8.4_

- [x] 11. Integration tests
  - [x] 11.1 Write integration tests (`tests/test_handler.py` or `tests/test_integration.py`)
    - Mock `WatchlistRunner` to return canned `TradingSignal` objects for 2 tickers
    - Verify full generator output sequence: button disabled → progress updates → signal cards → button re-enabled
    - Test multi-ticker partial failure: one ticker succeeds, one fails → verify both signal card and error card rendered
    - Test `ActivityFeedCallback` events translate to correct activity feed HTML entries
    - _Requirements: 3.1, 3.3, 5.1, 5.5, 8.1_

  - [x] 11.2 Create test fixtures (`tests/conftest.py`)
    - Define mock `TradingSignal` fixtures (BUY, SELL, HOLD examples)
    - Define mock `WatchlistRunner` that returns configurable results
    - Define mock `ActivityFeedCallback` event sequences
    - _Requirements: 5.1, 5.2_

- [x] 12. Final checkpoint - All tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document (10 properties total)
- Unit tests validate specific examples and edge cases
- The `crew/` package is imported as an external dependency — mock it in tests
- All code is Python targeting the Gradio 4.44.1 API

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3"] },
    { "id": 1, "tasks": ["2.1", "2.2", "4.1"] },
    {
      "id": 2,
      "tasks": ["2.3", "2.4", "2.5", "2.6", "4.2", "4.3", "4.4", "4.5", "4.6"]
    },
    { "id": 3, "tasks": ["4.7", "4.8", "4.9", "4.10", "6.1"] },
    { "id": 4, "tasks": ["6.2", "6.3", "6.4"] },
    { "id": 5, "tasks": ["7.1"] },
    { "id": 6, "tasks": ["7.2", "7.3", "7.4"] },
    { "id": 7, "tasks": ["7.5", "7.6", "9.1"] },
    { "id": 8, "tasks": ["10.1", "10.2", "10.3"] },
    { "id": 9, "tasks": ["11.1", "11.2"] }
  ]
}
```
