from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from runlet.core.agent import Agent
from runlet.core.messages import Message


def _message_list() -> list[Message]:
    return []


def _metadata_map() -> dict[str, Any]:
    return {}


def _state_map() -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    source: str = "actual"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class RunContext:
    run_id: str
    agent: Agent
    messages: list[Message] = field(default_factory=_message_list)
    metadata: dict[str, Any] = field(default_factory=_metadata_map)
    state: dict[str, Any] = field(default_factory=_state_map)
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: str
    output: str | None = None
    reasoning: str = ""
    messages: tuple[Message, ...] = ()
    usage: Usage = field(default_factory=Usage)
    error: str | None = None

    @classmethod
    def completed(
        cls,
        run_id: str,
        output: str,
        reasoning: str = "",
        usage: Usage | None = None,
        messages: tuple[Message, ...] = (),
    ) -> "RunResult":
        return cls(
            run_id=run_id,
            status="completed",
            output=output,
            reasoning=reasoning,
            usage=usage or Usage(),
            messages=messages,
        )

    @classmethod
    def failed(
        cls,
        run_id: str,
        error: str,
        reasoning: str = "",
        usage: Usage | None = None,
        messages: tuple[Message, ...] = (),
    ) -> "RunResult":
        return cls(
            run_id=run_id,
            status="failed",
            reasoning=reasoning,
            usage=usage or Usage(),
            error=error,
            messages=messages,
        )
