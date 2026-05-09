# Requirements Document

## Introduction

This document specifies the requirements for the FinAgent Gradio Frontend — a web-based user interface deployed as a Hugging Face Space. The frontend provides a financial terminal-style dashboard where users configure analysis parameters, observe real-time agent activity, and receive structured trading signals from the FinAgent multi-agent pipeline. The UI must handle long-running analysis gracefully and present results with clear visual hierarchy and color-coded trading actions.

## Glossary

- **Gradio_App**: The Gradio-based Python web application serving the FinAgent frontend interface
- **User_Input_Panel**: The UI section containing all input controls for configuring an analysis run
- **Agent_Activity_Feed**: The real-time scrolling log displaying agent execution status and intermediate outputs
- **Trading_Signals_Dashboard**: The output section displaying formatted signal cards for each analyzed ticker
- **Signal_Card**: A visual card component displaying the trading signal details for a single ticker
- **Ticker**: A stock or cryptocurrency symbol identifier (e.g., AAPL, BTC-USD)
- **Watchlist**: A comma-separated list of ticker symbols submitted by the user for analysis
- **Risk_Tolerance**: A user-selected parameter (Conservative, Moderate, Aggressive) that influences signal generation
- **Trading_Style**: A user-selected parameter (Day Trading, Swing Trading, Position Trading) that influences signal timeframes
- **Pipeline**: The agent-orchestration module's analysis workflow invoked by the frontend
- **HF_Space**: A Hugging Face Spaces deployment environment hosting the Gradio application
- **vLLM_Endpoint**: The environment-configurable URL for the vLLM inference server used by the agent pipeline

## Requirements

### Requirement 1: Ticker Input and Validation

**User Story:** As a trader, I want to enter a comma-separated list of ticker symbols, so that I can analyze multiple assets in a single run.

#### Acceptance Criteria

1. THE User_Input_Panel SHALL provide a text input field for entering comma-separated ticker symbols with placeholder text showing an example format (e.g., "AAPL, NVDA, TSLA, BTC-USD")
2. WHEN the user submits an empty ticker input, THE Gradio_App SHALL display a validation error message indicating that at least one ticker is required
3. WHEN the user submits ticker input, THE Gradio_App SHALL trim whitespace from each ticker symbol and convert symbols to uppercase
4. WHEN the user submits ticker input containing characters other than letters, digits, hyphens, and periods, THE Gradio_App SHALL display a validation error identifying the invalid characters
5. WHEN the user submits more than 10 tickers in a single watchlist, THE Gradio_App SHALL display a validation error indicating the maximum of 10 tickers per analysis run

### Requirement 2: Analysis Configuration Controls

**User Story:** As a trader, I want to configure risk tolerance, portfolio value, and trading style, so that the analysis is tailored to my investment profile.

#### Acceptance Criteria

1. THE User_Input_Panel SHALL provide a dropdown selector for Risk_Tolerance with options: Conservative, Moderate, Aggressive
2. THE User_Input_Panel SHALL provide a numeric input field for portfolio value with a default value of 10000 and a minimum value of 0
3. THE User_Input_Panel SHALL provide a dropdown selector for Trading_Style with options: Day Trading, Swing Trading, Position Trading
4. THE User_Input_Panel SHALL provide an "Analyze" button that triggers the analysis pipeline with the configured parameters
5. WHILE the Pipeline is executing, THE User_Input_Panel SHALL disable the "Analyze" button to prevent duplicate submissions

### Requirement 3: Real-Time Agent Activity Feed

**User Story:** As a user, I want to see which agent is currently running and its intermediate outputs, so that I understand the analysis progress and can verify the system is working.

#### Acceptance Criteria

1. WHEN the user clicks the "Analyze" button, THE Agent_Activity_Feed SHALL display a timestamped entry indicating analysis has started
2. WHEN an agent begins execution, THE Agent_Activity_Feed SHALL display the agent name and a visual spinner indicator
3. WHEN an agent produces intermediate output, THE Agent_Activity_Feed SHALL append a timestamped entry with the output content
4. WHEN an agent completes execution, THE Agent_Activity_Feed SHALL update the entry to show completion status and remove the spinner
5. THE Agent_Activity_Feed SHALL auto-scroll to the most recent entry as new entries are appended
6. WHILE the Pipeline is executing, THE Agent_Activity_Feed SHALL display elapsed time since analysis started

### Requirement 4: Progress Indication for Long-Running Analysis

**User Story:** As a user, I want clear progress indication during the 30-120 second analysis window, so that I know the system has not stalled.

#### Acceptance Criteria

1. WHEN the Pipeline begins execution, THE Gradio_App SHALL display a progress indicator showing that analysis is in progress
2. WHILE the Pipeline is executing, THE Gradio_App SHALL display the name of the ticker currently being analyzed
3. WHILE the Pipeline is executing for multiple tickers, THE Gradio_App SHALL display progress as "Analyzing ticker N of M"
4. IF the Pipeline execution exceeds 180 seconds without completing, THEN THE Gradio_App SHALL display a timeout warning message to the user

### Requirement 5: Trading Signals Dashboard Display

**User Story:** As a trader, I want to see structured signal cards for each ticker with clear BUY/SELL/HOLD actions, so that I can quickly assess trading opportunities.

#### Acceptance Criteria

1. WHEN the Pipeline completes analysis for a ticker, THE Trading_Signals_Dashboard SHALL display a Signal_Card containing: ticker name, current price, action (BUY/SELL/HOLD), confidence percentage, entry price, stop-loss price, target price, risk/reward ratio, and a brief reasoning summary
2. THE Signal_Card SHALL display BUY actions with green color coding, SELL actions with red color coding, and HOLD actions with yellow color coding
3. WHEN all tickers have been analyzed, THE Trading_Signals_Dashboard SHALL display an aggregate summary section showing total tickers analyzed, count of BUY signals, count of SELL signals, and count of HOLD signals
4. THE Signal_Card SHALL display the confidence percentage as a value between 0 and 100 with a visual indicator of confidence level
5. IF the Pipeline returns an error for a specific ticker, THEN THE Trading_Signals_Dashboard SHALL display an error card for that ticker indicating the failure reason

### Requirement 6: Dark Theme and Financial Terminal Aesthetic

**User Story:** As a trader, I want a dark-themed interface with financial terminal aesthetics, so that the tool feels professional and reduces eye strain during extended use.

#### Acceptance Criteria

1. THE Gradio_App SHALL use a dark color theme as the default and only theme
2. THE Gradio_App SHALL display a FinAgent title and branding in the header area
3. THE Gradio_App SHALL display a financial disclaimer footer stating that signals are for informational purposes only and do not constitute financial advice
4. THE Gradio_App SHALL use monospace or terminal-style fonts for numerical data and the Agent_Activity_Feed

### Requirement 7: Hugging Face Space Deployment

**User Story:** As a developer, I want to deploy the application to Hugging Face Spaces with minimal configuration, so that the demo is publicly accessible for the hackathon.

#### Acceptance Criteria

1. THE Gradio_App SHALL be structured as a single `app.py` entry point file compatible with Hugging Face Spaces Gradio SDK
2. THE Gradio_App SHALL read the vLLM inference endpoint URL from an environment variable named `VLLM_ENDPOINT_URL`
3. THE Gradio_App SHALL include a `requirements.txt` file listing all Python dependencies with pinned versions
4. IF the `VLLM_ENDPOINT_URL` environment variable is not set, THEN THE Gradio_App SHALL display a configuration error message indicating the missing endpoint
5. THE Gradio_App SHALL launch with `server_name="0.0.0.0"` to be accessible within the HF_Space container environment

### Requirement 8: Error Handling and Resilience

**User Story:** As a user, I want clear error messages when something goes wrong, so that I can understand failures and retry if appropriate.

#### Acceptance Criteria

1. IF the Pipeline raises an unhandled exception, THEN THE Gradio_App SHALL display a user-friendly error message in the Trading_Signals_Dashboard and re-enable the "Analyze" button
2. IF the connection to the vLLM_Endpoint fails, THEN THE Gradio_App SHALL display an error message indicating the inference service is unavailable
3. WHEN an error occurs during analysis, THE Agent_Activity_Feed SHALL display a timestamped error entry with the failure context
4. IF a network timeout occurs during Pipeline execution, THEN THE Gradio_App SHALL inform the user and suggest retrying with fewer tickers
