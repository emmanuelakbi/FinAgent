---
title: FinAgent
emoji: 🤖
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: Multi-agent trading signals powered by Qwen on AMD MI300X
tags:
  - crewai
  - qwen
  - amd-mi300x
  - trading
  - agents
---

# 🤖 FinAgent — AI-Powered Trading Signal Generator

**Built for the AMD Developer Hackathon** · Track: AI Agents & Agentic Workflows

FinAgent is a CrewAI-driven multi-agent system that analyzes any ticker symbol you throw at it — stocks, crypto, ETFs — through five specialized AI agents and returns a structured BUY / SELL / HOLD trading signal with confidence, entry price, stop loss, and target.

All the reasoning runs on **Qwen3-8B served by vLLM on an AMD Instinct MI300X** via the AMD Developer Cloud, accessed through a standard OpenAI-compatible API.

## The five agents

| Agent               | Role                                                | Tools                                           |
| ------------------- | --------------------------------------------------- | ----------------------------------------------- |
| Market Scanner      | Detect news, price changes, volume anomalies        | `search_news`, `get_price_change`, `get_volume` |
| Fundamental Analyst | Evaluate financials, earnings, peer comparison      | `get_financials`, `get_earnings`, `get_peers`   |
| Technical Analyst   | Read price history, indicators, entry / exit points | `get_price_history`, `calculate_indicators`     |
| Risk Manager        | Size the position, place ATR-based stop loss        | `calculate_position_size`, `set_stop_loss`      |
| Chief Strategist    | Synthesize all four into a final call               | — (pure reasoning)                              |

The first three run in parallel. Risk Manager waits for Technical Analyst's entry price. Chief Strategist waits for everyone.

## How it works

1. You paste a comma-separated watchlist (up to 10 tickers).
2. The Gradio UI streams agent activity live as they work.
3. Each ticker produces a signal card: action (BUY/SELL/HOLD), confidence, entry / stop / target prices, and a per-agent reasoning summary.
4. Failed tickers render an error card; one bad ticker doesn't stop the batch.

## Try it

Type something like:

```
AAPL, NVDA, BTC-USD, TSLA
```

pick your risk profile, portfolio value, and trading style, and hit **🔍 Analyze**.

## Architecture

```
┌──────────────────────────────┐  HTTPS   ┌─────────────────────────────┐
│  Hugging Face Space          │◄────────►│  AMD Developer Cloud        │
│  • Gradio 4.44 UI            │          │  • AMD Instinct MI300X      │
│  • CrewAI 1.14 orchestrator  │          │  • vLLM 0.6.3 + ROCm 6.2    │
│  • yfinance / ddgs tools     │          │  • Qwen/Qwen3-8B            │
└──────────────────────────────┘          └─────────────────────────────┘
```

## License

MIT — see [GitHub repository](https://github.com/your-username/FinAgent) for the full source, tests, and development plan (4 specs built with Kiro spec-driven development).

---

⚠️ **Disclaimer:** Trading signals produced here are for informational purposes only and do not constitute financial advice. Always do your own research before placing a trade.
