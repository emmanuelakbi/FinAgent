"""Integration tests for concurrent requests and latency on live vLLM server.

These tests are designed to run against a live vLLM server on MI300X hardware.
They validate latency SLAs, concurrent request handling, and response isolation.

Run with:
    cd inference/tests && python -m pytest test_integration_latency.py -v -m integration

Requirements validated: 6.1, 6.2, 6.3, 5.1, 5.2, 5.3
"""

import time
import json
import concurrent.futures

import pytest
import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SERVER_URL = "http://0.0.0.0:8000"
MODEL_NAME = "Qwen/Qwen3-8B"

# Generate ~1024 tokens of input by repeating a financial analysis prompt.
# Average English word is ~1.3 tokens; we use a dense financial prompt and repeat.
_BASE_PROMPT = (
    "Analyze the quarterly earnings report for the company. "
    "Consider revenue growth, operating margins, free cash flow, "
    "debt-to-equity ratio, and forward guidance. "
    "Evaluate the impact of macroeconomic conditions including "
    "interest rate policy, inflation trends, and consumer spending. "
    "Provide a detailed breakdown of segment performance, "
    "compare year-over-year and quarter-over-quarter metrics, "
    "and assess whether the current valuation is justified "
    "given the growth trajectory and risk factors. "
)

# Repeat to reach ~1024 tokens of input (~750 words ≈ 1024 tokens)
LONG_PROMPT = (_BASE_PROMPT * 15).strip()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server_url():
    """Return the vLLM server URL. Override via VLLM_SERVER_URL env var."""
    import os

    return os.environ.get("VLLM_SERVER_URL", DEFAULT_SERVER_URL)


@pytest.fixture(scope="module")
def model_name():
    """Return the model name to use in requests."""
    import os

    return os.environ.get("VLLM_MODEL_NAME", MODEL_NAME)


def _server_is_running(url: str) -> bool:
    """Check if the vLLM server is reachable."""
    try:
        resp = requests.get(f"{url}/v1/models", timeout=5)
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


@pytest.fixture(scope="module", autouse=True)
def skip_if_server_not_running(server_url):
    """Skip all tests in this module if the vLLM server is not running."""
    if not _server_is_running(server_url):
        pytest.skip(
            f"vLLM server not running at {server_url}. "
            "Start the server before running integration tests."
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _build_chat_payload(
    model: str, prompt: str, max_tokens: int = 512, stream: bool = False
) -> dict:
    """Build an OpenAI-compatible chat completion request payload."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": stream,
    }


def _send_request(server_url: str, payload: dict, timeout: float = 60) -> dict:
    """Send a chat completion request and return the response JSON."""
    resp = requests.post(
        f"{server_url}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestLatencyAndConcurrency:
    """Integration tests for latency SLAs and concurrent request handling."""

    def test_single_request_under_30_seconds(self, server_url, model_name):
        """Send a request with ~1024 input tokens and max_tokens=512.

        Verify the response arrives within 30 seconds.

        Validates: Requirements 6.1
        """
        payload = _build_chat_payload(model_name, LONG_PROMPT, max_tokens=512)

        start = time.perf_counter()
        response = _send_request(server_url, payload, timeout=30)
        elapsed = time.perf_counter() - start

        # Verify response structure
        assert "choices" in response, "Response missing 'choices' field"
        assert len(response["choices"]) > 0, "Response has empty choices"
        content = response["choices"][0]["message"]["content"]
        assert len(content) > 0, "Response content is empty"

        # Verify latency SLA
        assert elapsed < 30, (
            f"Single request took {elapsed:.2f}s, exceeds 30s SLA"
        )

    def test_five_concurrent_requests_under_60_seconds(
        self, server_url, model_name
    ):
        """Send 5 concurrent requests and verify all complete within 60 seconds.

        Validates: Requirements 6.2, 5.1, 5.2
        """
        prompts = [
            f"Analyze the financial performance of company #{i+1}. "
            f"Focus on revenue trends, profit margins, and market position. "
            f"Provide a comprehensive assessment. {_BASE_PROMPT}"
            for i in range(5)
        ]

        payloads = [
            _build_chat_payload(model_name, prompt, max_tokens=256)
            for prompt in prompts
        ]

        start = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(_send_request, server_url, payload, timeout=60)
                for payload in payloads
            ]
            results = [
                f.result() for f in concurrent.futures.as_completed(futures)
            ]

        elapsed = time.perf_counter() - start

        # Verify all 5 requests completed successfully
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"
        for i, result in enumerate(results):
            assert "choices" in result, f"Result {i} missing 'choices'"
            content = result["choices"][0]["message"]["content"]
            assert len(content) > 0, f"Result {i} has empty content"

        # Verify latency SLA
        assert elapsed < 60, (
            f"5 concurrent requests took {elapsed:.2f}s, exceeds 60s SLA"
        )

    def test_ttft_under_2_seconds(self, server_url, model_name):
        """Measure time to first token on an idle server. Verify < 2 seconds.

        Uses the streaming endpoint to measure TTFT accurately.

        Validates: Requirements 6.3
        """
        # Use a short prompt to measure TTFT on idle server
        prompt = "What is the current P/E ratio significance in stock valuation?"
        payload = _build_chat_payload(
            model_name, prompt, max_tokens=100, stream=True
        )

        start = time.perf_counter()
        ttft = None

        # Use streaming to measure time to first token
        with requests.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=10,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    decoded = line.decode("utf-8")
                    # SSE format: "data: {...}"
                    if decoded.startswith("data: ") and decoded != "data: [DONE]":
                        chunk = json.loads(decoded[6:])
                        # First chunk with content indicates first token
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            if delta.get("content"):
                                ttft = time.perf_counter() - start
                                break

        # If streaming didn't yield content deltas, fall back to first byte
        if ttft is None:
            # Fallback: measure time to first response byte (non-streaming)
            payload_non_stream = _build_chat_payload(
                model_name, prompt, max_tokens=100, stream=False
            )
            start = time.perf_counter()
            resp = requests.post(
                f"{server_url}/v1/chat/completions",
                json=payload_non_stream,
                timeout=10,
            )
            ttft = time.perf_counter() - start
            resp.raise_for_status()

        assert ttft is not None, "Could not measure TTFT"
        assert ttft < 2.0, (
            f"Time to first token was {ttft:.3f}s, exceeds 2s SLA"
        )

    def test_response_isolation(self, server_url, model_name):
        """Send 5 concurrent requests with distinct topics and verify isolation.

        Each request asks about a different stock ticker. Verify that each
        response relates to its own prompt and doesn't contain content
        from other prompts.

        Validates: Requirements 5.3
        """
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        prompts = [
            f"Tell me specifically about {ticker} stock. "
            f"Only discuss {ticker} and no other company. "
            f"What is {ticker}'s business model and recent performance?"
            for ticker in tickers
        ]

        payloads = [
            _build_chat_payload(model_name, prompt, max_tokens=200)
            for prompt in prompts
        ]

        # Track which payload maps to which ticker
        future_to_ticker = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for ticker, payload in zip(tickers, payloads):
                future = executor.submit(
                    _send_request, server_url, payload, timeout=60
                )
                future_to_ticker[future] = ticker

            results = {}
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                results[ticker] = future.result()

        # Verify each response relates to its own ticker
        other_tickers_map = {
            ticker: [t for t in tickers if t != ticker] for ticker in tickers
        }

        for ticker, response in results.items():
            content = response["choices"][0]["message"]["content"]
            assert len(content) > 0, f"Empty response for {ticker}"

            # The response should mention its own ticker
            assert ticker in content.upper() or ticker.lower() in content.lower(), (
                f"Response for {ticker} doesn't mention {ticker}. "
                f"Content: {content[:200]}..."
            )

            # The response should NOT prominently feature other tickers.
            # We check that other tickers don't appear as primary subjects.
            # A brief mention is acceptable (e.g., "unlike MSFT..."), but
            # the response shouldn't be primarily about another ticker.
            for other_ticker in other_tickers_map[ticker]:
                # Count occurrences — if another ticker appears more than
                # the requested ticker, that indicates content leakage
                own_count = content.upper().count(ticker)
                other_count = content.upper().count(other_ticker)
                assert other_count <= own_count, (
                    f"Response for {ticker} mentions {other_ticker} "
                    f"({other_count} times) more than {ticker} "
                    f"({own_count} times). Possible content leakage. "
                    f"Content: {content[:300]}..."
                )
