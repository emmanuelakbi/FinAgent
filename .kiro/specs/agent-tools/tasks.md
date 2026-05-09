# Implementation Plan: Agent Tools

## Overview

Implement the Python tool functions module for CrewAI agents in FinAgent. The module provides 10 tool functions across four domains (market scanning, fundamental analysis, technical analysis, risk management) with a shared TTL cache, utility helpers, and comprehensive error handling. All tools use keyless data sources (yfinance, duckduckgo-search, pandas-ta) and return formatted strings.

## Tasks

- [x] 1. Set up project structure and dependencies
  - [x] 1.1 Create module directory structure and `__init__.py`
    - Create `tools/` directory with `__init__.py`, `cache.py`, `utils.py`, `market_scanner.py`, `fundamental_analyst.py`, `technical_analyst.py`, `risk_manager.py`
    - `__init__.py` should re-export all 10 `@tool` decorated functions
    - Create `tests/` directory with `test_cache.py`, `test_utils.py`, `test_market_scanner.py`, `test_fundamental.py`, `test_technical.py`, `test_risk_manager.py`, `test_error_handling.py`, and `tests/integration/test_live_apis.py`
    - _Requirements: 1.1, 1.5_

  - [x] 1.2 Add project dependencies
    - Add `crewai[tools]`, `yfinance`, `duckduckgo-search`, `pandas-ta` as runtime dependencies
    - Add `pytest`, `hypothesis`, `pytest-mock`, `freezegun` as dev/test dependencies
    - Create or update `requirements.txt` or `pyproject.toml` with pinned versions
    - _Requirements: 1.1_

- [x] 2. Implement cache module
  - [x] 2.1 Implement `TTLCache` class in `tools/cache.py`
    - Implement `CacheEntry` dataclass with `value: str` and `timestamp: float`
    - Implement `TTLCache.__init__` with `default_ttl=300` (5 min) and `max_age=900` (15 min)
    - Implement `make_key(func_name, **kwargs)` using `f"{func_name}:{sorted_params_json}"` for deterministic keys
    - Implement `get(key)` that returns cached value if within TTL, else None; triggers `_evict_stale()`
    - Implement `set(key, value)` that stores value with current timestamp
    - Implement `_evict_stale()` that removes entries older than `max_age` (15 min)
    - Implement `clear()` for test support
    - Use `threading.Lock` for thread safety on all read/write operations
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 2.2 Write property test: Cache round-trip within TTL
    - **Property 2: Cache round-trip within TTL**
    - Generate random (key, value) pairs; store and retrieve within TTL; verify exact match
    - **Validates: Requirements 3.1**

  - [x] 2.3 Write property test: Cache expiry after TTL
    - **Property 3: Cache expiry after TTL**
    - Generate random entries; freeze time past 5 minutes; verify `get` returns None
    - Use `freezegun` to manipulate time
    - **Validates: Requirements 3.2**

  - [x] 2.4 Write property test: Cache key determinism and uniqueness
    - **Property 4: Cache key determinism and uniqueness**
    - Generate random (func_name, params) tuples; verify same inputs → same key, different inputs → different keys
    - **Validates: Requirements 3.3**

  - [x] 2.5 Write property test: Cache eviction of stale entries
    - **Property 5: Cache eviction of stale entries**
    - Generate entries with old timestamps (>15 min); trigger cache access; verify stale entries removed
    - **Validates: Requirements 3.5**

- [x] 3. Implement utilities module
  - [x] 3.1 Implement utility functions in `tools/utils.py`
    - Implement `validate_ticker(ticker: str) -> tuple[bool, str]` that strips whitespace, rejects empty/whitespace-only strings with error message, and returns uppercase normalized ticker
    - Implement `format_currency(value: float, precision: int = 2) -> str` that formats with B/M/K units
    - Implement `safe_get(info: dict, key: str, default: str = "N/A") -> str` that returns default for None/missing
    - Implement `format_percent(value: float, precision: int = 2) -> str` that formats with % sign
    - _Requirements: 4.3, 4.4, 8.3_

  - [x] 3.2 Write property test: Ticker normalization to uppercase
    - **Property 6: Ticker normalization to uppercase**
    - Generate random non-whitespace strings; verify `validate_ticker` returns uppercase
    - **Validates: Requirements 4.3**

  - [x] 3.3 Write property test: Whitespace and empty ticker rejection
    - **Property 7: Whitespace and empty ticker rejection**
    - Generate whitespace-only strings (including empty); verify return contains "Error"
    - **Validates: Requirements 4.4, 2.1**

- [x] 4. Checkpoint - Ensure cache and utility tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement market scanner tools
  - [x] 5.1 Implement `search_news` in `tools/market_scanner.py`
    - Use `DDGS().news(keywords=ticker, max_results=5, timelimit="w")` for 7-day window
    - Format output with up to 5 numbered articles (title + snippet)
    - Handle empty results with "no recent news" message
    - Follow the standard error handling pattern (validate → cache check → API call → format → cache store)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 1.1, 1.2, 1.3, 2.5_

  - [x] 5.2 Implement `get_price_change` in `tools/market_scanner.py`
    - Use `yfinance.Ticker(ticker).info` for `currentPrice` and `previousClose`
    - Fall back to `history(period="2d")` if info fields unavailable
    - Calculate percentage: `((current - previous) / previous) * 100` rounded to 2 decimals
    - Format output with current price, previous close, absolute change, and percentage change
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 1.1, 1.2, 1.3, 2.5_

  - [x] 5.3 Implement `get_volume` in `tools/market_scanner.py`
    - Use `history(period="25d")` to get 20+ days of volume data
    - Calculate 20-day average volume and volume ratio rounded to 2 decimals
    - Include "UNUSUAL VOLUME" flag when ratio > 2.0
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 1.1, 1.2, 1.3, 2.5_

  - [x] 5.4 Write property test: News results bounded and date-filtered
    - **Property 8: News results bounded and date-filtered**
    - Generate lists of 0-20 news items with random dates; verify output has ≤5 items all within 7 days
    - **Validates: Requirements 5.2, 5.3**

  - [x] 5.5 Write property test: Price change percentage formula correctness
    - **Property 9: Price change percentage formula correctness**
    - Generate random (current_price, previous_close) pairs where previous_close > 0; verify percentage formula
    - **Validates: Requirements 6.3**

  - [x] 5.6 Write property test: Volume ratio computation and UNUSUAL VOLUME flag
    - **Property 10: Volume ratio computation and UNUSUAL VOLUME flag**
    - Generate random (current_volume, avg_volume) pairs where avg_volume > 0; verify ratio and flag presence
    - **Validates: Requirements 7.2, 7.3**

  - [x] 5.7 Write unit tests for market scanner tools
    - Test `search_news` format with 3 mocked results and empty results
    - Test `get_price_change` with positive and negative price changes
    - Test `get_volume` with normal volume (no flag) and unusual volume (flag present)
    - _Requirements: 5.2, 5.4, 6.2, 7.2, 7.3_

- [x] 6. Implement fundamental analyst tools
  - [x] 6.1 Implement `get_financials` in `tools/fundamental_analyst.py`
    - Use `yfinance.Ticker(ticker).info` for marketCap, trailingPE, revenueGrowth, profitMargins, debtToEquity
    - Format monetary values with `format_currency`, percentages with `format_percent`
    - Mark unavailable metrics as "N/A" using `safe_get` (no error for missing fields)
    - _Requirements: 8.1, 8.2, 8.3, 1.1, 1.2, 1.3, 2.5_

  - [x] 6.2 Implement `get_earnings` in `tools/fundamental_analyst.py`
    - Use `yfinance.Ticker(ticker).earnings_dates` or `.quarterly_earnings` for last 4 quarters
    - Calculate surprise: `((reported - estimated) / |estimated|) * 100` rounded to 2 decimals
    - Return "not available for this instrument type" for crypto tickers
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 1.1, 1.2, 1.3, 2.5_

  - [x] 6.3 Implement `get_peers` in `tools/fundamental_analyst.py`
    - Retrieve sector/industry from `ticker.info` and identify up to 5 peer companies
    - Return "not available for this instrument type" for crypto/ETFs
    - _Requirements: 10.1, 10.2, 10.3, 1.1, 1.2, 1.3, 2.5_

  - [x] 6.4 Write property test: Graceful N/A for missing financial metrics
    - **Property 11: Graceful N/A for missing financial metrics**
    - Generate random subsets of 5 metrics as None; verify "N/A" appears for missing, no "Error" in output
    - **Validates: Requirements 8.3**

  - [x] 6.5 Write property test: Earnings surprise percentage formula correctness
    - **Property 12: Earnings surprise percentage formula correctness**
    - Generate random (reported_EPS, estimated_EPS) pairs where estimated ≠ 0; verify surprise formula
    - **Validates: Requirements 9.3**

  - [x] 6.6 Write unit tests for fundamental analyst tools
    - Test `get_financials` with all fields present (mock complete info dict)
    - Test `get_earnings` with crypto ticker returning "not available" message
    - Test `get_peers` with crypto ticker returning "not available" message
    - _Requirements: 8.2, 9.4, 10.3_

- [x] 7. Checkpoint - Ensure all market scanner and fundamental tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement technical analyst tools
  - [x] 8.1 Implement `get_price_history` in `tools/technical_analyst.py`
    - Use `yfinance.Ticker(ticker).history(period="90d")` to ensure 60 trading days
    - Calculate indicators via pandas-ta: RSI(14), MACD(12/26/9), SMA(20), SMA(50), BBands(20, 2)
    - Return last 5 days of computed data in tabular format
    - Mark indicators as "N/A" if insufficient data (< 50 days)
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 1.1, 1.2, 1.3, 2.5_

  - [x] 8.2 Implement `calculate_indicators` in `tools/technical_analyst.py`
    - Compute current RSI, MACD, and Bollinger Band values
    - Classify RSI: < 30 → "BUY", > 70 → "SELL", else → "NEUTRAL"
    - Classify MACD: bullish crossover → "BUY", bearish crossover → "SELL", else → "NEUTRAL"
    - Classify Bollinger: close < lower → "BUY", close > upper → "SELL", else → "NEUTRAL"
    - Return each indicator's value and signal classification
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 1.1, 1.2, 1.3, 2.5_

  - [x] 8.3 Write property test: Signal classification from indicator values
    - **Property 13: Signal classification from indicator values**
    - Generate random RSI values in [0,100], MACD tuples, and Bollinger tuples; verify signal classification logic
    - **Validates: Requirements 12.2, 12.3, 12.4**

  - [x] 8.4 Write unit tests for technical analyst tools
    - Test `get_price_history` with insufficient data (30 days) showing "N/A" for SMA50
    - Test `calculate_indicators` with all-neutral scenario (RSI=50, no crossover, price in bands)
    - _Requirements: 11.4, 12.5_

- [x] 9. Implement risk manager tools
  - [x] 9.1 Implement `calculate_position_size` in `tools/risk_manager.py`
    - Validate: portfolio_value > 0, entry_price > 0, 0 < risk_percent ≤ 100, entry_price ≠ stop_loss
    - Formula: `shares = floor((portfolio_value * risk_percent / 100) / |entry_price - stop_loss|)`
    - Return shares (whole number), dollar amount at risk, and total position value
    - This tool does NOT use cache (pure computation)
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 1.1, 1.2, 1.3, 2.5_

  - [x] 9.2 Implement `set_stop_loss` in `tools/risk_manager.py`
    - Validate: atr_multiplier > 0
    - Retrieve 14-period ATR via `pandas_ta.atr(high, low, close, length=14)`
    - Calculate: `stop_loss = entry_price - (ATR * atr_multiplier)`, `take_profit = entry_price + (ATR * atr_multiplier * 2)`
    - Return entry price, ATR value, stop-loss, take-profit, and risk-reward ratio (1:2)
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 1.1, 1.2, 1.3, 2.5_

  - [x] 9.3 Write property test: Position size formula correctness
    - **Property 14: Position size formula correctness**
    - Generate random valid inputs (portfolio_value > 0, 0 < risk_percent ≤ 100, entry_price > 0, stop_loss ≠ entry_price); verify shares formula and derived values
    - **Validates: Requirements 13.1, 13.2**

  - [x] 9.4 Write property test: Risk manager input validation
    - **Property 15: Risk manager input validation**
    - Generate invalid inputs (negative risk%, zero portfolio, equal entry/stop, negative multiplier); verify "Error" in output
    - **Validates: Requirements 13.3, 13.4, 13.5, 14.4**

  - [x] 9.5 Write property test: Stop-loss and take-profit formula correctness
    - **Property 16: Stop-loss and take-profit formula correctness**
    - Generate random valid (entry_price > 0, ATR > 0, atr_multiplier > 0); verify stop-loss and take-profit formulas yield 1:2 risk-reward
    - **Validates: Requirements 14.2, 14.3**

  - [x] 9.6 Write unit tests for risk manager tools
    - Test `calculate_position_size` basic: $100K portfolio, 1% risk, $50 entry, $48 stop → 500 shares
    - Test `set_stop_loss` basic: entry $100, ATR $5, multiplier 1.5 → SL $92.50, TP $115.00
    - _Requirements: 13.2, 14.3_

- [x] 10. Implement cross-cutting error handling tests
  - [x] 10.1 Write property test: Tool functions never raise exceptions
    - **Property 1: Tool functions never raise exceptions**
    - Mock internals of each tool to raise random exceptions (network errors, invalid data, runtime errors); verify no exception propagates and return contains "Error"
    - Test all 10 tool functions
    - **Validates: Requirements 2.5, 2.1, 2.2, 2.3**

- [x] 11. Checkpoint - Ensure all property and unit tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Wire module together and write integration tests
  - [x] 12.1 Finalize `tools/__init__.py` exports and verify module imports
    - Ensure all 10 tool functions are importable from `tools` package
    - Verify `from tools import search_news, get_price_change, get_volume, get_financials, get_earnings, get_peers, get_price_history, calculate_indicators, calculate_position_size, set_stop_loss` works
    - Verify each function has `@tool` decorator and proper docstring
    - _Requirements: 1.1, 1.5_

  - [x] 12.2 Write integration tests against live APIs
    - Test `search_news("AAPL")` returns non-error response
    - Test `get_price_change("NVDA")` and `get_price_change("BTC-USD")` return price data
    - Test `get_volume("AAPL")` returns volume data
    - Test `get_financials("AAPL")` returns all 5 metrics
    - Test `get_earnings("AAPL")` returns 4 quarters
    - Test `get_peers("AAPL")` returns sector and peer list
    - Test `get_price_history("AAPL")` returns indicators
    - Test `calculate_indicators("AAPL")` returns all 3 signals
    - Test `set_stop_loss("AAPL", 185.0, 1.5)` returns levels
    - Mark all with `@pytest.mark.integration`
    - _Requirements: 4.1, 4.2, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1, 14.1_

- [x] 13. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document (16 properties total)
- Unit tests validate specific examples and edge cases
- Integration tests require network access and may be rate-limited — run separately from CI
- All tools use Python with yfinance, duckduckgo-search, pandas-ta, and Hypothesis for property testing
- The `calculate_position_size` tool is pure computation and does not use the cache

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1", "3.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "2.4", "2.5", "3.2", "3.3"] },
    {
      "id": 3,
      "tasks": ["5.1", "5.2", "5.3", "6.1", "6.2", "6.3", "9.1", "9.2"]
    },
    {
      "id": 4,
      "tasks": [
        "5.4",
        "5.5",
        "5.6",
        "5.7",
        "6.4",
        "6.5",
        "6.6",
        "8.1",
        "8.2",
        "9.3",
        "9.4",
        "9.5",
        "9.6"
      ]
    },
    { "id": 5, "tasks": ["8.3", "8.4", "10.1"] },
    { "id": 6, "tasks": ["12.1"] },
    { "id": 7, "tasks": ["12.2"] }
  ]
}
```
