"""LLM configuration, timeouts, and constants for the orchestration layer."""

from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """Configuration for the vLLM endpoint connection."""

    base_url: str = "http://localhost:8000/v1"
    model_name: str = "Qwen/Qwen3-14B"
    temperature: float = 0.7
    max_tokens: int = 1024
    request_timeout: int = 120  # seconds


@dataclass
class CrewConfig:
    """Configuration for crew execution parameters."""

    max_iterations: int = 5
    task_timeout: int = 120  # seconds per task
    verbose: bool = True


@dataclass
class OrchestratorConfig:
    """Top-level configuration combining all settings."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    crew: CrewConfig = field(default_factory=CrewConfig)
