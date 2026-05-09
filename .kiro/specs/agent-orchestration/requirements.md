# Requirements Document

## Introduction

This module defines the CrewAI multi-agent orchestration layer for FinAgent — an autonomous financial research and trading signal system. The orchestration layer coordinates five specialized AI agents (Market Scanner, Fundamental Analyst, Technical Analyst, Risk Manager, and Chief Strategist) to analyze tickers from a watchlist and produce structured trading signals (BUY/SELL/HOLD) with confidence percentages. All agents use a local vLLM endpoint via OpenAI-compatible API, and the system supports multi-ticker analysis with real-time activity logging for the Gradio UI.

## Glossary

- **Crew**: A CrewAI Crew instance that orchestrates the execution of all agent tasks for a single ticker analysis run
- **Agent**: A CrewAI Agent configured with a role, goal, backstory, LLM connection, and optional tools
- **Task**: A CrewAI Task assigned to a specific Agent, with defined expected output and optional dependencies
- **Market_Scanner_Agent**: Agent responsible for monitoring news and detecting significant market events for a given ticker
- **Fundamental_Analyst_Agent**: Agent responsible for analyzing company financials and determining intrinsic value
- **Technical_Analyst_Agent**: Agent responsible for analyzing price charts and technical indicators to identify entry/exit points
- **Risk_Manager_Agent**: Agent responsible for calculating position sizes and setting stop-loss levels
- **Chief_Strategist_Agent**: Agent responsible for synthesizing all other agent outputs into a final trading signal; uses no tools, only reasoning
- **Trading_Signal**: The structured output containing ticker, action (BUY/SELL/HOLD), confidence percentage, entry price, stop-loss, target price, and reasoning summaries
- **Watchlist**: A comma-separated list of ticker symbols to analyze
- **vLLM_Endpoint**: The local vLLM inference server exposing an OpenAI-compatible API at a configurable base_url
- **Activity_Feed**: A real-time log of agent actions and outputs streamed to the Gradio UI via callbacks
- **Max_Iterations**: The maximum number of reasoning loops an agent may perform before being forced to produce output (set to 5)
- **Task_Timeout**: The maximum wall-clock time allowed for a single agent task before it is terminated (set to 120 seconds)

## Requirements

### Requirement 1: Agent LLM Configuration

**User Story:** As a developer, I want all agents to connect to the local vLLM endpoint via an OpenAI-compatible base_url, so that inference runs on the local AMD GPU without external API dependencies.

#### Acceptance Criteria

1. THE Crew SHALL configure each Agent with an LLM connection pointing to the vLLM_Endpoint using an OpenAI-compatible base_url
2. WHEN the vLLM_Endpoint base_url is changed in configuration, THE Crew SHALL use the updated base_url for all Agent LLM calls without code changes
3. IF the vLLM_Endpoint is unreachable during Agent initialization, THEN THE Crew SHALL raise a descriptive connection error within 10 seconds

### Requirement 2: Market Scanner Agent Definition

**User Story:** As a trader, I want a Market Scanner Agent that monitors news and detects significant market events, so that I am alerted to material developments affecting my watchlist.

#### Acceptance Criteria

1. THE Market_Scanner_Agent SHALL be configured with the role "Market Scanner", a goal to detect significant market events, and access to the search_news, get_price_change, and get_volume tools
2. WHEN assigned a ticker, THE Market_Scanner_Agent SHALL produce output containing identified market events, price change magnitude, and volume anomalies
3. WHILE executing, THE Market_Scanner_Agent SHALL complete within the Task_Timeout of 120 seconds
4. WHILE executing, THE Market_Scanner_Agent SHALL complete within Max_Iterations of 5 reasoning loops

### Requirement 3: Fundamental Analyst Agent Definition

**User Story:** As a trader, I want a Fundamental Analyst Agent that evaluates company financials, so that I understand the intrinsic value of a stock before trading.

#### Acceptance Criteria

1. THE Fundamental_Analyst_Agent SHALL be configured with the role "Fundamental Analyst", a goal to determine intrinsic value, and access to the get_financials, get_earnings, and get_peers tools
2. WHEN assigned a ticker, THE Fundamental_Analyst_Agent SHALL produce output containing a valuation assessment with supporting financial metrics
3. WHILE executing, THE Fundamental_Analyst_Agent SHALL complete within the Task_Timeout of 120 seconds
4. WHILE executing, THE Fundamental_Analyst_Agent SHALL complete within Max_Iterations of 5 reasoning loops

### Requirement 4: Technical Analyst Agent Definition

**User Story:** As a trader, I want a Technical Analyst Agent that analyzes price charts and indicators, so that I can identify optimal entry and exit points.

#### Acceptance Criteria

1. THE Technical_Analyst_Agent SHALL be configured with the role "Technical Analyst", a goal to identify entry/exit points, and access to the get_price_history and calculate_indicators tools
2. WHEN assigned a ticker, THE Technical_Analyst_Agent SHALL produce output containing a recommended entry price, target price, and supporting technical indicator values
3. WHILE executing, THE Technical_Analyst_Agent SHALL complete within the Task_Timeout of 120 seconds
4. WHILE executing, THE Technical_Analyst_Agent SHALL complete within Max_Iterations of 5 reasoning loops

### Requirement 5: Risk Manager Agent Definition

**User Story:** As a trader, I want a Risk Manager Agent that calculates position sizes and stop-losses, so that my capital is protected on every trade.

#### Acceptance Criteria

1. THE Risk_Manager_Agent SHALL be configured with the role "Risk Manager", a goal to protect capital through position sizing and stop-loss placement, and access to the calculate_position_size and set_stop_loss tools
2. WHEN provided with the Technical_Analyst_Agent output containing an entry price, THE Risk_Manager_Agent SHALL produce output containing a position size, stop-loss price, and risk-reward ratio
3. WHILE executing, THE Risk_Manager_Agent SHALL complete within the Task_Timeout of 120 seconds
4. WHILE executing, THE Risk_Manager_Agent SHALL complete within Max_Iterations of 5 reasoning loops

### Requirement 6: Chief Strategist Agent Definition

**User Story:** As a trader, I want a Chief Strategist Agent that synthesizes all analysis into a final trading signal, so that I receive a single actionable recommendation.

#### Acceptance Criteria

1. THE Chief_Strategist_Agent SHALL be configured with the role "Chief Strategist", a goal to synthesize all agent outputs into a final trading signal, and no tools (pure reasoning only)
2. WHEN provided with outputs from all other agents, THE Chief_Strategist_Agent SHALL produce a Trading_Signal containing the ticker, action (BUY, SELL, or HOLD), and a confidence percentage between 0 and 100
3. THE Chief_Strategist_Agent SHALL include in the Trading_Signal an entry price, stop-loss price, and target price derived from the Risk_Manager_Agent and Technical_Analyst_Agent outputs
4. THE Chief_Strategist_Agent SHALL include in the Trading_Signal a reasoning summary from each contributing agent
5. WHILE executing, THE Chief_Strategist_Agent SHALL complete within the Task_Timeout of 120 seconds
6. WHILE executing, THE Chief_Strategist_Agent SHALL complete within Max_Iterations of 5 reasoning loops

### Requirement 7: Task Dependency and Execution Order

**User Story:** As a developer, I want tasks to execute with correct dependencies, so that agents receive the inputs they need and the pipeline runs efficiently.

#### Acceptance Criteria

1. THE Crew SHALL execute the Market_Scanner_Agent task, Fundamental_Analyst_Agent task, and Technical_Analyst_Agent task in parallel (no dependencies between them)
2. THE Crew SHALL execute the Risk_Manager_Agent task only after the Technical_Analyst_Agent task has completed successfully
3. THE Crew SHALL execute the Chief_Strategist_Agent task only after all other agent tasks have completed
4. IF a predecessor task fails, THEN THE Crew SHALL skip dependent tasks and report the failure in the final output

### Requirement 8: Agent Failure Handling

**User Story:** As a developer, I want the crew to handle agent failures gracefully, so that a single agent failure does not crash the entire pipeline.

#### Acceptance Criteria

1. IF an Agent exceeds the Task_Timeout of 120 seconds, THEN THE Crew SHALL terminate that Agent task and record a timeout failure
2. IF an Agent exceeds Max_Iterations of 5, THEN THE Crew SHALL force the Agent to produce its best available output
3. IF an Agent produces unparseable output, THEN THE Crew SHALL record the failure with the raw output and continue execution of independent tasks
4. WHEN a failure occurs, THE Crew SHALL include the failure details in the Activity_Feed log

### Requirement 9: Structured Trading Signal Output

**User Story:** As a trader, I want the final output in a consistent structured format, so that downstream systems (Gradio UI, alerts) can parse and display it reliably.

#### Acceptance Criteria

1. THE Chief_Strategist_Agent SHALL produce output conforming to the Trading_Signal format: "TICKER — ACTION (Confidence: XX%)" followed by entry price, stop-loss, target, and reasoning summaries
2. THE Trading_Signal action field SHALL contain exactly one of: BUY, SELL, or HOLD
3. THE Trading_Signal confidence field SHALL be an integer between 0 and 100 inclusive
4. IF the Chief_Strategist_Agent output does not conform to the Trading_Signal format, THEN THE Crew SHALL attempt to parse the output into the correct format using a fallback parser

### Requirement 10: Multi-Ticker Watchlist Support

**User Story:** As a trader, I want to analyze multiple tickers from my watchlist in a single run, so that I can review signals for my entire portfolio efficiently.

#### Acceptance Criteria

1. WHEN provided a Watchlist containing comma-separated ticker symbols, THE Crew SHALL run the full analysis pipeline for each ticker in sequence
2. THE Crew SHALL aggregate all Trading_Signals into a single results collection upon completion of all tickers
3. IF a single ticker analysis fails, THEN THE Crew SHALL continue processing remaining tickers and include the failure in the aggregated results
4. THE Crew SHALL log the start and completion of each ticker analysis to the Activity_Feed

### Requirement 11: Real-Time Activity Feed Logging

**User Story:** As a user viewing the Gradio UI, I want to see real-time updates of agent activity, so that I can monitor the analysis progress and understand what each agent is doing.

#### Acceptance Criteria

1. THE Crew SHALL invoke a callback function at the start and completion of each Agent task, passing the agent name, task status, and timestamp
2. WHEN an Agent produces intermediate output, THE Crew SHALL invoke the callback with the agent name and a summary of the output
3. THE Crew SHALL invoke the callback when a ticker analysis begins and when it completes
4. THE Activity_Feed callback interface SHALL accept a structured payload containing: event_type, agent_name, ticker, message, and timestamp fields
