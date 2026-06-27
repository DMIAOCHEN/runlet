from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class StateScope:
    kind: str
    key: str


class StateStore(Protocol):
    async def load(self, scope: StateScope) -> dict[str, Any]:
        ...

    async def save(self, scope: StateScope, state: dict[str, Any]) -> None:
        ...


class InMemoryStateStore:
    def __init__(self) -> None:
        self._states: dict[StateScope, dict[str, Any]] = {}

    async def load(self, scope: StateScope) -> dict[str, Any]:
        return dict(self._states.get(scope, {}))

    async def save(self, scope: StateScope, state: dict[str, Any]) -> None:
        self._states[scope] = dict(state)
