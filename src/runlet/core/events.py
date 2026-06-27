from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _event_map() -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class RuntimeEvent:
    type: str
    run_id: str
    id: str = field(default_factory=lambda: f"evt_{uuid4().hex}")
    timestamp: datetime = field(default_factory=utc_now)
    step_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    agent_name: str | None = None
    severity: str = "info"
    payload: dict[str, Any] = field(default_factory=_event_map)
    attributes: dict[str, Any] = field(default_factory=_event_map)


class EventSink(Protocol):
    async def emit(self, event: RuntimeEvent) -> None:
        ...


class InMemoryObserver:
    def __init__(self) -> None:
        self.events: list[RuntimeEvent] = []

    async def emit(self, event: RuntimeEvent) -> None:
        self.events.append(event)


class CompositeEventSink:
    def __init__(self, sinks: list[EventSink] | tuple[EventSink, ...]) -> None:
        self.sinks = tuple(sinks)

    async def emit(self, event: RuntimeEvent) -> None:
        for sink in self.sinks:
            await sink.emit(event)
