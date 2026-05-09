# Requirements Document

## Introduction

This module covers the Python tool functions that CrewAI agents use in FinAgent — an autonomous multi-agent financial research and trading signal system. Each tool is a standalone function decorated with CrewAI's `@tool` decorator, returning string outputs that agents can consume. The tools span four domains: market scanning (news and price data), fundamental analysis (financials and earnings), technical analysis (indicators and signals), and risk management (position sizing and stop-loss). All tools use free, keyless data sources (yfinance, duckduckgo-search, pandas-ta) and must handle errors gracefully to keep agent workflows uninterrupted.

## Glossary

- **Tool_Function**: A Python function decorated with CrewAI's `@tool` decorator that accepts typed parameters and returns a string result
- **Market_Scanner**: The set of Tool_Functions that retrieve recent news, price changes, and volume data for a given ticker
- **Fundamental_Analyst**: The set of Tool_Functions that retrieve financial metrics, earnings data, and sector peers for a given ticker
- **Technical_Analyst**: The set of Tool_Functions that retrieve price history with indicators and calculate buy/sell signals for a given ticker
- **Risk_Manager**: The set of Tool_Functions that calculate position sizing and stop-loss/take-profit levels
- **Ticker**: A string identifier for a financial instrument, including stock symbols (AAPL, NVDA) and crypto pairs (BTC-USD, ETH-USD)
- **Cache**: An in-memory time-based cache that stores API responses to avoid redundant requests and rate limiting
- **Indicator_Parameters**: The standard technical indicator settings: RSI 14-period, MACD 12/26/9, SMA 20-period, SMA 50-period, Bollinger Bands 20-period with 2 standard deviations
- **Signal**: A string classification of a technical indicator reading as "BUY", "SELL", or "NEUTRAL"

## Requirements

### Requirement 1: Tool Function Structure and CrewAI Compatibility

**User Story:** As a CrewAI agent developer, I want all tool functions to follow CrewAI's tool format, so that agents can invoke them and parse their outputs consistently.

#### Acceptance Criteria

1. THE Tool_Function SHALL be decorated with CrewAI's `@tool` decorator and include a docstring describing the tool's purpose and parameters
2. THE Tool_Function SHALL accept only typed parameters (str, float, int) and return a single string value
3. WHEN a Tool_Function completes successfully, THE Tool_Function SHALL return a human-readable string containing the requested data formatted with labeled fields
4. WHEN a Tool_Function completes successfully, THE Tool_Function SHALL return the result within 30 seconds of invocation
5. THE Tool_Function SHALL be independently importable and callable without requiring initialization of other Tool_Functions or shared mutable state beyond the Cache

### Requirement 2: Error Handling Across All Tools

**User Story:** As a CrewAI agent developer, I want tools to handle errors gracefully, so that agent workflows continue even when data sources are unavailable.

#### Acceptance Criteria

1. IF a Tool_Function receives a Ticker that does not correspond to a valid financial instrument in the data source, THEN THE Tool_Function SHALL return a string containing the phrase "Error" and a message indicating the Ticker was not found
2. IF a Tool_Function encounters a network timeout or connection failure when contacting an external data source, THEN THE Tool_Function SHALL return a string containing the phrase "Error" and a message indicating the data source is unreachable
3. IF a Tool_Function encounters rate limiting from an external data source, THEN THE Tool_Function SHALL return a string containing the phrase "Error" and a message indicating rate limiting occurred
4. IF a Tool_Function receives data from an external source with missing or null fields required for computation, THEN THE Tool_Function SHALL return a string containing the phrase "Error" and a message indicating which data fields are unavailable
5. THE Tool_Function SHALL catch all exceptions internally and return an error string rather than raising an unhandled exception to the calling agent

### Requirement 3: Response Caching

**User Story:** As a developer, I want tool results cached with a time-to-live, so that repeated agent queries within a short window do not trigger redundant API calls or rate limiting.

#### Acceptance Criteria

1. WHEN a Tool_Function is called with a Ticker that has a cached result less than 5 minutes old, THE Tool_Function SHALL return the cached result without making an external API call
2. WHEN a Tool_Function is called with a Ticker that has no cached result or a cached result older than 5 minutes, THE Tool_Function SHALL make a fresh external API call and store the result in the Cache
3. THE Cache SHALL use the combination of function name and all input parameters as the cache key to distinguish between different queries
4. THE Cache SHALL store results in memory without requiring external infrastructure (no Redis, no database)
5. WHILE the Cache contains entries older than 15 minutes, THE Cache SHALL evict those entries on the next cache access to prevent unbounded memory growth

### Requirement 4: Ticker Compatibility

**User Story:** As a CrewAI agent developer, I want tools to work with both stock tickers and crypto pairs, so that agents can analyze any supported financial instrument.

#### Acceptance Criteria

1. WHEN a Tool_Function receives a stock Ticker (such as AAPL, NVDA, AMD, TSLA), THE Tool_Function SHALL retrieve and return data for that stock
2. WHEN a Tool_Function receives a crypto Ticker (such as BTC-USD, ETH-USD), THE Tool_Function SHALL retrieve and return data for that cryptocurrency pair
3. WHEN a Tool_Function receives a Ticker, THE Tool_Function SHALL normalize the Ticker to uppercase before querying external data sources
4. IF a Tool_Function receives an empty string or a string containing only whitespace as a Ticker, THEN THE Tool_Function SHALL return a string containing the phrase "Error" and a message indicating an invalid ticker was provided

### Requirement 5: Search News Tool

**User Story:** As a market scanning agent, I want to search recent news for a ticker, so that I can identify market-moving events and sentiment.

#### Acceptance Criteria

1. WHEN search_news is called with a valid Ticker, THE Market_Scanner SHALL query DuckDuckGo for recent news articles related to that Ticker
2. WHEN search_news retrieves results, THE Market_Scanner SHALL return a string containing up to 5 news items, each including the article title and a brief snippet or description
3. WHEN search_news retrieves results, THE Market_Scanner SHALL return only articles from the most recent 7 days
4. IF search_news finds no news articles for the given Ticker, THEN THE Market_Scanner SHALL return a string indicating no recent news was found for that Ticker

### Requirement 6: Get Price Change Tool

**User Story:** As a market scanning agent, I want to get the recent price change for a ticker, so that I can identify stocks with significant price movement.

#### Acceptance Criteria

1. WHEN get_price_change is called with a valid Ticker, THE Market_Scanner SHALL retrieve the current price and the price from the previous trading day using yfinance
2. WHEN get_price_change retrieves price data, THE Market_Scanner SHALL return a string containing the current price, the previous close price, the absolute price change, and the percentage change
3. WHEN get_price_change calculates the percentage change, THE Market_Scanner SHALL compute it as ((current_price - previous_close) / previous_close) \* 100 rounded to 2 decimal places
4. IF get_price_change cannot retrieve price data because the market is closed and no recent data exists, THEN THE Market_Scanner SHALL return the most recent available closing price with a note indicating the market status

### Requirement 7: Get Volume Tool

**User Story:** As a market scanning agent, I want to get volume compared to average, so that I can identify unusual trading activity.

#### Acceptance Criteria

1. WHEN get_volume is called with a valid Ticker, THE Market_Scanner SHALL retrieve the current day's trading volume and the 20-day average volume using yfinance
2. WHEN get_volume retrieves volume data, THE Market_Scanner SHALL return a string containing the current volume, the 20-day average volume, and the volume ratio (current divided by average) rounded to 2 decimal places
3. WHEN the volume ratio exceeds 2.0, THE Market_Scanner SHALL include the label "UNUSUAL VOLUME" in the returned string
4. IF get_volume is called outside of trading hours, THEN THE Market_Scanner SHALL use the most recent completed trading day's volume as the current volume

### Requirement 8: Get Financials Tool

**User Story:** As a fundamental analysis agent, I want to get key financial metrics for a ticker, so that I can assess the company's financial health.

#### Acceptance Criteria

1. WHEN get_financials is called with a valid Ticker, THE Fundamental_Analyst SHALL retrieve financial data using yfinance including market capitalization, P/E ratio, revenue growth, profit margin, and debt-to-equity ratio
2. WHEN get_financials retrieves data, THE Fundamental_Analyst SHALL return a string containing each metric with its label and value, with monetary values formatted with appropriate units (B for billions, M for millions)
3. IF one or more financial metrics are unavailable for the given Ticker (common for crypto or newly listed companies), THEN THE Fundamental_Analyst SHALL return the available metrics and mark unavailable metrics as "N/A" rather than returning an error

### Requirement 9: Get Earnings Tool

**User Story:** As a fundamental analysis agent, I want to get recent earnings data, so that I can evaluate earnings performance and surprises.

#### Acceptance Criteria

1. WHEN get_earnings is called with a valid Ticker, THE Fundamental_Analyst SHALL retrieve the most recent 4 quarters of earnings data using yfinance
2. WHEN get_earnings retrieves data, THE Fundamental_Analyst SHALL return a string containing for each quarter: the quarter date, the reported EPS, the estimated EPS, and the surprise percentage
3. WHEN get_earnings calculates the surprise percentage, THE Fundamental_Analyst SHALL compute it as ((reported_EPS - estimated_EPS) / |estimated_EPS|) \* 100 rounded to 2 decimal places
4. IF earnings data is unavailable for the given Ticker (common for crypto), THEN THE Fundamental_Analyst SHALL return a string indicating earnings data is not available for that instrument type

### Requirement 10: Get Peers Tool

**User Story:** As a fundamental analysis agent, I want to get sector peers for comparison, so that I can contextualize a company's performance relative to competitors.

#### Acceptance Criteria

1. WHEN get_peers is called with a valid Ticker, THE Fundamental_Analyst SHALL retrieve the sector and industry classification and identify up to 5 peer companies in the same sector using yfinance
2. WHEN get_peers retrieves peer data, THE Fundamental_Analyst SHALL return a string containing the sector name, industry name, and a list of peer tickers with their company names
3. IF peer data is unavailable for the given Ticker (common for crypto or ETFs), THEN THE Fundamental_Analyst SHALL return a string indicating peer comparison is not available for that instrument type

### Requirement 11: Get Price History with Technical Indicators

**User Story:** As a technical analysis agent, I want price history with calculated indicators, so that I can analyze trends and momentum.

#### Acceptance Criteria

1. WHEN get_price_history is called with a valid Ticker, THE Technical_Analyst SHALL retrieve 60 trading days of daily OHLCV (open, high, low, close, volume) data using yfinance
2. WHEN get_price_history retrieves price data, THE Technical_Analyst SHALL calculate RSI using a 14-period window, MACD using 12/26/9 parameters, SMA using 20-period and 50-period windows, and Bollinger Bands using a 20-period window with 2 standard deviations, all computed via pandas-ta
3. WHEN get_price_history completes calculations, THE Technical_Analyst SHALL return a string containing the most recent 5 days of data with date, close price, RSI, MACD value, MACD signal, SMA20, SMA50, upper Bollinger Band, and lower Bollinger Band
4. IF fewer than 50 trading days of data are available for the given Ticker, THEN THE Technical_Analyst SHALL calculate only the indicators that have sufficient data and mark others as "N/A"

### Requirement 12: Calculate Buy/Sell Signals

**User Story:** As a technical analysis agent, I want buy/sell signals derived from indicators, so that I can provide actionable trading recommendations.

#### Acceptance Criteria

1. WHEN calculate_indicators is called with a valid Ticker, THE Technical_Analyst SHALL compute the current RSI, MACD, and Bollinger Band values using the Indicator_Parameters
2. WHEN RSI is below 30, THE Technical_Analyst SHALL classify the RSI Signal as "BUY" (oversold); WHEN RSI is above 70, THE Technical_Analyst SHALL classify the RSI Signal as "SELL" (overbought); otherwise THE Technical_Analyst SHALL classify the RSI Signal as "NEUTRAL"
3. WHEN the MACD line crosses above the MACD signal line (current MACD > signal AND previous MACD <= signal), THE Technical_Analyst SHALL classify the MACD Signal as "BUY"; WHEN the MACD line crosses below the signal line, THE Technical_Analyst SHALL classify the MACD Signal as "SELL"; otherwise THE Technical_Analyst SHALL classify the MACD Signal as "NEUTRAL"
4. WHEN the current close price is below the lower Bollinger Band, THE Technical_Analyst SHALL classify the Bollinger Signal as "BUY"; WHEN the current close price is above the upper Bollinger Band, THE Technical_Analyst SHALL classify the Bollinger Signal as "SELL"; otherwise THE Technical_Analyst SHALL classify the Bollinger Signal as "NEUTRAL"
5. WHEN calculate_indicators completes, THE Technical_Analyst SHALL return a string containing each indicator's current value and its corresponding Signal classification

### Requirement 13: Calculate Position Size

**User Story:** As a risk management agent, I want to calculate safe position sizes, so that I can recommend trades that respect portfolio risk limits.

#### Acceptance Criteria

1. WHEN calculate_position_size is called with portfolio_value, risk_percent, entry_price, and stop_loss, THE Risk_Manager SHALL compute the position size as (portfolio_value \* risk_percent / 100) / |entry_price - stop_loss|
2. WHEN calculate_position_size completes, THE Risk_Manager SHALL return a string containing the number of shares (rounded down to a whole number), the dollar amount at risk, and the total position value
3. IF risk_percent is less than 0 or greater than 100, THEN THE Risk_Manager SHALL return a string containing the phrase "Error" and a message indicating risk_percent must be between 0 and 100
4. IF entry_price equals stop_loss, THEN THE Risk_Manager SHALL return a string containing the phrase "Error" and a message indicating entry price and stop loss cannot be equal
5. IF portfolio_value is less than or equal to 0 or entry_price is less than or equal to 0, THEN THE Risk_Manager SHALL return a string containing the phrase "Error" and a message indicating the value must be positive

### Requirement 14: Set Stop-Loss and Take-Profit Levels

**User Story:** As a risk management agent, I want suggested stop-loss and take-profit levels based on volatility, so that I can recommend protective exit points.

#### Acceptance Criteria

1. WHEN set_stop_loss is called with entry_price and atr_multiplier, THE Risk_Manager SHALL retrieve the 14-period Average True Range (ATR) for the instrument using yfinance and pandas-ta
2. WHEN set_stop_loss computes levels, THE Risk_Manager SHALL calculate stop_loss as entry_price minus (ATR _ atr_multiplier) and take_profit as entry_price plus (ATR _ atr_multiplier \* 2), both rounded to 2 decimal places
3. WHEN set_stop_loss completes, THE Risk_Manager SHALL return a string containing the entry price, the ATR value, the stop-loss price, the take-profit price, and the risk-reward ratio
4. IF atr_multiplier is less than or equal to 0, THEN THE Risk_Manager SHALL return a string containing the phrase "Error" and a message indicating the ATR multiplier must be positive
5. IF ATR data is unavailable due to insufficient price history, THEN THE Risk_Manager SHALL return a string containing the phrase "Error" and a message indicating insufficient data to calculate ATR
