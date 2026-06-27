from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _string_map() -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class Message:
    role: str
    text: str
    metadata: dict[str, Any] = field(default_factory=_string_map)

    @classmethod
    def system(cls, text: str, metadata: dict[str, Any] | None = None) -> "Message":
        return cls(role="system", text=text, metadata=metadata or {})

    @classmethod
    def user(cls, text: str, metadata: dict[str, Any] | None = None) -> "Message":
        return cls(role="user", text=text, metadata=metadata or {})

    @classmethod
    def assistant(cls, text: str, metadata: dict[str, Any] | None = None) -> "Message":
        return cls(role="assistant", text=text, metadata=metadata or {})

    @classmethod
    def tool(cls, text: str, metadata: dict[str, Any] | None = None) -> "Message":
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
