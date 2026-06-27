from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _metadata_map() -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class Agent:
    name: str
    instructions: str
    model: Any
    tools: tuple[Any, ...] = ()
    hooks: tuple[Any, ...] = ()
    metadata: dict[str, Any] = field(default_factory=_metadata_map)
