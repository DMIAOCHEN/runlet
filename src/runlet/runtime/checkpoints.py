from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from runlet.core.human import HumanRequest
from runlet.core.messages import Message, ToolCall
from runlet.core.models import ModelRequest
from runlet.core.runs import Usage


def _metadata_map() -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class RunCheckpoint:
    id: str
    run_id: str
    agent_name: str
    request: ModelRequest
    messages: tuple[Message, ...]
    pending_request: HumanRequest | None
    pending_tool_call: ToolCall | None
    step: int
    reasoning: str
    usage: Usage
    pending_tool_calls: tuple[ToolCall, ...] = ()
    metadata: dict[str, Any] = field(default_factory=_metadata_map)


class CheckpointStore(Protocol):
    async def save(self, checkpoint: RunCheckpoint) -> None:
        ...

    async def load(self, checkpoint_id: str) -> RunCheckpoint | None:
        ...

    async def delete(self, checkpoint_id: str) -> None:
        ...


class InMemoryCheckpointStore:
    def __init__(self) -> None:
        self._checkpoints: dict[str, RunCheckpoint] = {}

    async def save(self, checkpoint: RunCheckpoint) -> None:
        self._checkpoints[checkpoint.id] = checkpoint

    async def load(self, checkpoint_id: str) -> RunCheckpoint | None:
        return self._checkpoints.get(checkpoint_id)

    async def delete(self, checkpoint_id: str) -> None:
        self._checkpoints.pop(checkpoint_id, None)
