import unittest

from runlet import Agent, Runtime
from runlet.events import InMemoryObserver
from runlet.testing import FakeStreamingModelProvider


class RuntimeStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_stream_yields_model_deltas(self) -> None:
        model = FakeStreamingModelProvider(["hel", "lo"])
        observer = InMemoryObserver()
        runtime = Runtime(event_sink=observer)
        agent = Agent(name="assistant", instructions="Help.", model=model)

        deltas: list[str] = []
        async for event in runtime.stream(agent, "hi"):
            if event.type == "model.stream.delta":
                deltas.append(event.payload["delta"])

        self.assertEqual(deltas, ["hel", "lo"])
        self.assertIn("model.stream.started", [event.type for event in observer.events])
        self.assertIn("model.stream.completed", [event.type for event in observer.events])
