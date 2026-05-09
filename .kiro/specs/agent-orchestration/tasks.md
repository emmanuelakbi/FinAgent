# Implementation Plan: Agent Orchestration

## Overview

This plan implements the CrewAI multi-agent orchestration layer for FinAgent. The implementation proceeds bottom-up: configuration and data models first, then individual modules (signals, callbacks, agents, tasks), then the orchestrator (crew), and finally the runner. Property-based tests validate correctness properties from the design, and unit/integration tests ensure end-to-end behavior.

## Tasks

- [x] 1. Set up project structure and dependencies
  - [x] 1.1 Create the `crew/` package directory structure with `__init__.py`
    - Create `crew/__init__.py` with placeholder re-exports for `FinAgentCrew` and `WatchlistRunner`
    - Create empty module files: `config.py`, `agents.py`, `tasks.py`, `crew.py`, `callbacks.py`, `signals.py`, `runner.py`
    - _Requirements: 1.1, 1.2_

  - [x] 1.2 Add CrewAI and langchain-openai dependencies
    - Add `crewai`, `langchain-openai` to project dependencies (pyproject.toml or requirements.txt)
    - Add `hypothesis` and `pytest` as dev dependencies for property-based and unit testing
    - Verify imports resolve correctly
    - _Requirements: 1.1_

- [x] 2. Implement configuration module
  - [x] 2.1 Implement `crew/config.py` with dataclass configuration hierarchy
    - Implement `LLMConfig` dataclass with fields: `base_url`, `model_name`, `temperature`, `max_tokens`, `request_timeout`
    - Implement `CrewConfig` dataclass with fields: `max_iterations`, `task_timeout`, `verbose`
    - Implement `OrchestratorConfig` dataclass composing `LLMConfig` and `CrewConfig` with defaults
    - _Requirements: 1.1, 1.2_

- [x] 3. Implement signals module
  - [x] 3.1 Implement `crew/signals.py` with `Action` enum and `TradingSignal` dataclass
    - Define `Action` enum with BUY, SELL, HOLD values
    - Define `TradingSignal` dataclass with fields: `ticker`, `action`, `confidence`, `entry_price`, `stop_loss`, `target_price`, `reasoning`
    - Implement `validate_confidence` static method that clamps values to [0, 100]
    - _Requirements: 9.1, 9.2, 9.3_

  - [x] 3.2 Implement `TradingSignalParser` with primary and fallback parsing
    - Implement `PRIMARY_PATTERN` regex for structured format: `"TICKER — ACTION (Confidence: XX%)"`
    - Implement `_parse_primary` method using the primary regex
    - Implement `ACTION_PATTERN`, `CONFIDENCE_PATTERN`, `PRICE_PATTERN` fallback regexes
    - Implement `_parse_fallback` method for heuristic extraction
    - Implement `_extract_prices` to pull entry, stop-loss, target from `$XX.XX` patterns
    - Implement `_extract_reasoning` to pull per-agent reasoning summaries
    - Implement `parse` method that tries primary then fallback
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 8.3_

  - [x] 3.3 Write property test: Trading signal round-trip parsing
    - **Property 2: Round trip consistency**
    - Generate arbitrary valid TradingSignal instances, format to expected string, parse back, assert equality of ticker, action, confidence, entry_price, stop_loss, target_price
    - **Validates: Requirements 6.2, 6.3, 9.1**

  - [x] 3.4 Write property test: Action field validation
    - **Property 3: Action field validation**
    - Generate strings that do NOT contain exactly one of BUY/SELL/HOLD as a standalone word, assert parser returns None or a valid Action enum
    - **Validates: Requirements 9.2**

  - [x] 3.5 Write property test: Confidence clamping
    - **Property 4: Confidence clamping to valid range**
    - Generate arbitrary integers, assert `validate_confidence` returns value in [0, 100], values < 0 become 0, values > 100 become 100
    - **Validates: Requirements 9.3**

  - [x] 3.6 Write property test: Fallback parser extraction
    - **Property 5: Fallback parser extraction**
    - Generate strings containing one of BUY/SELL/HOLD and a number followed by %, assert fallback parser extracts correct action and confidence in [0, 100]
    - **Validates: Requirements 9.4**

  - [x] 3.7 Write property test: Unparseable output yields None
    - **Property 6: Unparseable output yields None**
    - Generate strings containing none of BUY/SELL/HOLD, assert parser returns None
    - **Validates: Requirements 8.3**

- [x] 4. Implement callbacks module
  - [x] 4.1 Implement `crew/callbacks.py` with `EventType`, `ActivityEvent`, and `ActivityFeedCallback`
    - Define `EventType` enum with values: TICKER_START, TICKER_COMPLETE, TASK_START, TASK_COMPLETE, TASK_FAILED, AGENT_OUTPUT, CREW_ERROR
    - Define `ActivityEvent` dataclass with fields: `event_type`, `agent_name`, `ticker`, `message`, `timestamp`
    - Implement `ActivityFeedCallback` class with methods: `on_ticker_start`, `on_ticker_complete`, `on_task_start`, `on_task_complete`, `on_task_failed`, `on_agent_output`, `_emit`
    - Each method constructs an `ActivityEvent` and dispatches via `_emit` to the registered handler
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

  - [x] 4.2 Write property test: ActivityEvent structural completeness
    - **Property 9: ActivityEvent structural completeness**
    - Generate arbitrary valid EventType, non-empty agent_name, non-empty ticker, message, and datetime; construct ActivityEvent; assert all fields match inputs exactly
    - **Validates: Requirements 11.1, 11.4**

- [x] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement agents module
  - [x] 6.1 Implement `crew/agents.py` with `create_llm` and all 5 agent factory functions
    - Implement `create_llm(config: LLMConfig) -> ChatOpenAI` that creates a shared LLM instance pointing to the vLLM endpoint
    - Implement `create_market_scanner(llm, tools)` returning an Agent with role "Market Scanner", correct goal/backstory, and tools `[search_news, get_price_change, get_volume]`
    - Implement `create_fundamental_analyst(llm, tools)` returning an Agent with role "Fundamental Analyst", correct goal/backstory, and tools `[get_financials, get_earnings, get_peers]`
    - Implement `create_technical_analyst(llm, tools)` returning an Agent with role "Technical Analyst", correct goal/backstory, and tools `[get_price_history, calculate_indicators]`
    - Implement `create_risk_manager(llm, tools)` returning an Agent with role "Risk Manager", correct goal/backstory, and tools `[calculate_position_size, set_stop_loss]`
    - Implement `create_chief_strategist(llm)` returning an Agent with role "Chief Strategist", correct goal/backstory, and empty tools list
    - All agents configured with `max_iter=5` and `verbose=True`
    - _Requirements: 1.1, 2.1, 2.3, 2.4, 3.1, 3.3, 3.4, 4.1, 4.3, 4.4, 5.1, 5.3, 5.4, 6.1, 6.5, 6.6_

  - [x] 6.2 Write property test: LLM base_url propagation
    - **Property 1: LLM base_url propagation**
    - Generate arbitrary valid URL strings as `base_url` in LLMConfig, create LLM via `create_llm`, assert the resulting ChatOpenAI instance has `base_url` matching the input
    - **Validates: Requirements 1.2**

- [x] 7. Implement tasks module
  - [x] 7.1 Implement `crew/tasks.py` with all task factory functions
    - Implement `create_market_scan_task(agent, ticker)` returning a Task with proper description and expected_output
    - Implement `create_fundamental_task(agent, ticker)` returning a Task with proper description and expected_output
    - Implement `create_technical_task(agent, ticker)` returning a Task with proper description and expected_output
    - Implement `create_risk_task(agent, ticker, context)` returning a Task with `context=[technical_task]` dependency
    - Implement `create_strategy_task(agent, ticker, context)` returning a Task with `context=[market_task, fundamental_task, technical_task, risk_task]` dependency
    - Strategy task description includes the exact output format specification
    - _Requirements: 7.1, 7.2, 7.3, 2.2, 3.2, 4.2, 5.2, 6.2, 6.3, 6.4_

  - [x] 7.2 Write unit tests for task factory functions
    - Test that each task factory returns a Task with correct agent assignment
    - Test that `create_risk_task` has context containing the technical task
    - Test that `create_strategy_task` has context containing all 4 predecessor tasks
    - Test that task descriptions include the ticker symbol
    - _Requirements: 7.1, 7.2, 7.3_

- [x] 8. Implement crew module
  - [x] 8.1 Implement `crew/crew.py` with `CrewResult` dataclass and `FinAgentCrew` class
    - Define `CrewResult` dataclass with fields: `ticker`, `signal`, `raw_output`, `success`, `error`
    - Implement `FinAgentCrew.__init__` accepting `config`, `tools` dict, and optional `callback`
    - Implement `FinAgentCrew._build_crew(ticker)` that creates agents, tasks with dependencies, and returns a configured `Crew` instance
    - Implement `FinAgentCrew._parse_output(raw_output, ticker)` that delegates to `TradingSignalParser`
    - Implement `FinAgentCrew.run(ticker)` that builds crew, executes with error handling (timeout, agent errors), parses output, and returns `CrewResult`
    - Emit callback events at task start/complete/failure points
    - _Requirements: 1.1, 7.1, 7.2, 7.3, 7.4, 8.1, 8.2, 8.3, 8.4, 9.1, 9.4_

  - [x] 8.2 Write unit tests for FinAgentCrew
    - Test that `_build_crew` creates correct number of agents and tasks
    - Test that `_parse_output` delegates to TradingSignalParser correctly
    - Test that `run` returns CrewResult with `success=False` and error message on exception
    - Mock CrewAI Crew execution to test error handling paths
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [x] 9. Implement runner module
  - [x] 9.1 Implement `crew/runner.py` with `WatchlistResult` dataclass and `WatchlistRunner` class
    - Define `WatchlistResult` dataclass with fields: `signals`, `total_tickers`, `successful`, `failed`
    - Implement `WatchlistRunner.__init__` accepting `config`, `tools`, and optional `callback`
    - Implement `WatchlistRunner._parse_watchlist(watchlist)` that splits on commas, strips whitespace, uppercases, and removes empty entries
    - Implement `WatchlistRunner._run_single(ticker)` that creates a `FinAgentCrew` and calls `run(ticker)` with error isolation (try/except)
    - Implement `WatchlistRunner.run(watchlist)` that parses watchlist, iterates tickers, emits callbacks, aggregates results into `WatchlistResult`
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [x] 9.2 Write property test: Watchlist parsing produces normalized ticker list
    - **Property 7: Watchlist parsing produces normalized ticker list**
    - Generate comma-separated strings with arbitrary whitespace and mixed case, assert `_parse_watchlist` returns uppercase trimmed list with correct count
    - **Validates: Requirements 10.1**

  - [x] 9.3 Write property test: WatchlistResult aggregation invariant
    - **Property 8: WatchlistResult aggregation invariant**
    - Generate lists of CrewResult objects with random success/failure, construct WatchlistResult, assert `total_tickers == len(list)`, `successful + failed == total_tickers`
    - **Validates: Requirements 10.2**

  - [x] 9.4 Write unit tests for WatchlistRunner
    - Test that a failed ticker does not prevent subsequent tickers from running
    - Test that callbacks are emitted for ticker start and complete
    - Test that WatchlistResult counts match actual outcomes
    - Mock FinAgentCrew to simulate success and failure scenarios
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [x] 10. Wire up `crew/__init__.py` re-exports
  - [x] 10.1 Update `crew/__init__.py` with final public API exports
    - Re-export: `FinAgentCrew`, `WatchlistRunner`, `OrchestratorConfig`, `LLMConfig`, `CrewConfig`, `TradingSignal`, `Action`, `CrewResult`, `WatchlistResult`, `ActivityFeedCallback`, `ActivityEvent`, `EventType`
    - _Requirements: 1.1, 1.2_

- [x] 11. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Integration wiring and final validation
  - [x] 12.1 Write integration tests for the full pipeline
    - Test end-to-end flow: config → agents → tasks → crew → runner with mocked LLM responses
    - Test that a multi-ticker watchlist produces correct WatchlistResult structure
    - Test that callback events are emitted in correct order (ticker_start → task_start → task_complete → ticker_complete)
    - Test graceful degradation: one ticker fails, others succeed
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 8.1, 8.3, 10.2, 10.3, 11.1, 11.3_

- [x] 13. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- All code is Python, using CrewAI framework with langchain-openai for LLM connectivity
- The `tools` dict passed to FinAgentCrew maps agent names to tool function lists from the `agent-tools` module

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["3.1", "4.1"] },
    { "id": 3, "tasks": ["3.2", "4.2"] },
    { "id": 4, "tasks": ["3.3", "3.4", "3.5", "3.6", "3.7"] },
    { "id": 5, "tasks": ["6.1"] },
    { "id": 6, "tasks": ["6.2", "7.1"] },
    { "id": 7, "tasks": ["7.2", "8.1"] },
    { "id": 8, "tasks": ["8.2", "9.1"] },
    { "id": 9, "tasks": ["9.2", "9.3", "9.4", "10.1"] },
    { "id": 10, "tasks": ["12.1"] }
  ]
}
```
