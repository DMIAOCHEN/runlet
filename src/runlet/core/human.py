from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from runlet.core.messages import ToolCall


def _metadata_map() -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class HumanOption:
    id: str
    label: str
    description: str | None = None


@dataclass(frozen=True)
class HumanRequest:
    id: str
    kind: Literal["tool_approval", "choice", "input"]
    prompt: str
    options: tuple[HumanOption, ...] = ()
    tool_call: ToolCall | None = None
    metadata: dict[str, Any] = field(default_factory=_metadata_map)


@dataclass(frozen=True)
class HumanResponse:
    request_id: str
    action: Literal["approve", "reject", "select", "submit"]
    value: str | None = None
    edited_arguments: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=_metadata_map)
