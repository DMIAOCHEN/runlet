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


class ModelProvider(Protocol):
    async def complete(self, request: ModelRequest) -> ModelResponse:
        ...

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        ...

    async def capabilities(self) -> ModelCapabilities:
        ...
