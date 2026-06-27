from dataclasses import dataclass


@dataclass(frozen=True)
class RunPolicy:
    max_steps: int = 20


@dataclass(frozen=True)
class ContextPolicy:
    reserved_output_tokens: int = 1024
    max_context_tokens: int | None = None
    overflow_behavior: str = "fail"


@dataclass(frozen=True)
class HookPolicy:
    error_behavior: str = "fail"


@dataclass(frozen=True)
class ToolPolicy:
    timeout_seconds: float | None = None
