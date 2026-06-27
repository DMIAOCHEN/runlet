from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from runlet.core.messages import Message, ToolCall
from runlet.core.runs import Usage


def _tool_list() -> list[Any]:
    return []


def _metadata_map() -> dict[str, Any]:
    return {}


def _tool_call_list() -> list[ToolCall]:
    return []


@dataclass(frozen=True)
class ModelCapabilities:
    model_name: str
    context_window: int
    supports_tools: bool = True
    supports_parallel_tool_calls: bool = False
    supports_streaming: bool = False


@dataclass(frozen=True)
class ModelRequest:
    messages: list[Message]
    tools: list[Any] = field(default_factory=_tool_list)
    metadata: dict[str, Any] = field(default_factory=_metadata_map)
    options: dict[str, Any] = field(default_factory=_metadata_map)


@dataclass(frozen=True)
class ModelResponse:
    message: Message
    tool_calls: list[ToolCall] = field(default_factory=_tool_call_list)
    usage: Usage = field(default_factory=Usage)
    final: bool = True
    raw: Any = None


@dataclass(frozen=True)
class ModelStreamEvent:
    delta: str = ""
    tool_call: ToolCall | None = None
    usage: Usage | None = None
    final: bool = False
    raw: Any = None


def _arguments_map() -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class ProviderStreamEvent:
    kind: str
    delta: str = ""
    call_id: str | None = None
    name: str | None = None
    arguments_delta: str = ""
    arguments: dict[str, Any] = field(default_factory=_arguments_map)
    usage: Usage | None = None
    raw: Any = None

    @classmethod
    def text_delta(cls, delta: str, raw: Any = None) -> "ProviderStreamEvent":
        return cls(kind="text_delta", delta=delta, raw=raw)

    @classmethod
    def tool_call_delta(
        cls,
        call_id: str,
        name: str | None,
        arguments_delta: str,
        raw: Any = None,
    ) -> "ProviderStreamEvent":
        return cls(
            kind="tool_call_delta",
            call_id=call_id,
            name=name,
            arguments_delta=arguments_delta,
            raw=raw,
        )

    @classmethod
    def tool_call_completed(
        cls,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
        raw: Any = None,
    ) -> "ProviderStreamEvent":
        return cls(
            kind="tool_call_completed",
            call_id=call_id,
            name=name,
            arguments=dict(arguments),
            raw=raw,
        )

    @classmethod
    def message_completed(cls, raw: Any = None) -> "ProviderStreamEvent":
        return cls(kind="message_completed", raw=raw)

    @classmethod
    def usage_event(cls, usage: Usage, raw: Any = None) -> "ProviderStreamEvent":
        return cls(kind="usage", usage=usage, raw=raw)

    @classmethod
    def completed(cls, raw: Any = None) -> "ProviderStreamEvent":
        return cls(kind="completed", raw=raw)


class ModelProvider(Protocol):
    async def complete(self, request: ModelRequest) -> ModelResponse:
        ...

    async def stream(self, request: ModelRequest) -> AsyncIterator[ProviderStreamEvent | ModelStreamEvent]:
        ...

    async def capabilities(self) -> ModelCapabilities:
        ...
