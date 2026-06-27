from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _string_map() -> dict[str, Any]:
    return {}


def _message_list() -> list["Message"]:
    return []


def _state_map() -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class Message:
    role: str
    text: str
    metadata: dict[str, Any] = field(default_factory=_string_map)

    @classmethod
    def system(cls, text: str, metadata: dict[str, Any] | None = None) -> Message:
        return cls(role="system", text=text, metadata=metadata or {})

    @classmethod
    def user(cls, text: str, metadata: dict[str, Any] | None = None) -> Message:
        return cls(role="user", text=text, metadata=metadata or {})

    @classmethod
    def assistant(cls, text: str, metadata: dict[str, Any] | None = None) -> Message:
        return cls(role="assistant", text=text, metadata=metadata or {})

    @classmethod
    def tool(cls, text: str, metadata: dict[str, Any] | None = None) -> Message:
        return cls(role="tool", text=text, metadata=metadata or {})


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=_string_map)
    metadata: dict[str, Any] = field(default_factory=_string_map)


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    content: str
    metadata: dict[str, Any] = field(default_factory=_string_map)


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    source: str = "actual"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class Agent:
    name: str
    instructions: str
    model: Any
    tools: tuple[Any, ...] = ()
    hooks: tuple[Any, ...] = ()
    metadata: dict[str, Any] = field(default_factory=_string_map)


@dataclass
class RunContext:
    run_id: str
    agent: Agent
    messages: list[Message] = field(default_factory=_message_list)
    metadata: dict[str, Any] = field(default_factory=_string_map)
    state: dict[str, Any] = field(default_factory=_state_map)
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: str
    output: str | None = None
    messages: tuple[Message, ...] = ()
    usage: Usage = field(default_factory=Usage)
    error: str | None = None

    @classmethod
    def completed(
        cls,
        run_id: str,
        output: str,
        usage: Usage | None = None,
        messages: tuple[Message, ...] = (),
    ) -> RunResult:
        return cls(
            run_id=run_id,
            status="completed",
            output=output,
            usage=usage or Usage(),
            messages=messages,
        )

    @classmethod
    def failed(
        cls,
        run_id: str,
        error: str,
        usage: Usage | None = None,
        messages: tuple[Message, ...] = (),
    ) -> RunResult:
        return cls(
            run_id=run_id,
            status="failed",
            usage=usage or Usage(),
            error=error,
            messages=messages,
        )
