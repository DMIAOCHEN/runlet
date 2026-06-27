import unittest

from runlet.core.events import CompositeEventSink, InMemoryObserver, RuntimeEvent


class EventTests(unittest.IsolatedAsyncioTestCase):
    async def test_in_memory_observer_records_events(self) -> None:
        observer = InMemoryObserver()
        event = RuntimeEvent(type="run.started", run_id="run_1", payload={"input": "hello"})

        await observer.emit(event)

        self.assertEqual(len(observer.events), 1)
        self.assertEqual(observer.events[0].type, "run.started")
        self.assertEqual(observer.events[0].payload["input"], "hello")

    async def test_composite_sink_fans_out_events(self) -> None:
        first = InMemoryObserver()
        second = InMemoryObserver()
        sink = CompositeEventSink([first, second])
        event = RuntimeEvent(type="run.completed", run_id="run_1")

        await sink.emit(event)

        self.assertEqual(first.events[0].type, "run.completed")
        self.assertEqual(second.events[0].type, "run.completed")
