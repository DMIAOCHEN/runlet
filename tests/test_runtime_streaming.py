import unittest

from runlet import Agent, Runtime
from runlet.core import Message
from runlet.core.events import InMemoryObserver
from runlet.core.models import ModelRequest, ProviderStreamEvent
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

    async def test_runtime_stream_request_preserves_request_options(self) -> None:
        model = FakeStreamingModelProvider(["hel", "lo"])
        runtime = Runtime()
        agent = Agent(name="assistant", instructions="Help.", model=model)
        request = ModelRequest(
            messages=[Message.system("Override system."), Message.user("hi")],
            options={"openai": {"extra_body": {"store": False}}},
        )

        deltas: list[str] = []
        async for event in runtime.stream_request(agent, request):
            if event.type == "model.stream.delta":
                deltas.append(event.payload["delta"])

        self.assertEqual(deltas, ["hel", "lo"])
        self.assertEqual(
            model.requests[0].options,
            {"openai": {"extra_body": {"store": False}}},
        )
        self.assertEqual(model.requests[0].messages[0].text, "Override system.")

    async def test_runtime_stream_yields_reasoning_deltas_and_completed_payload(self) -> None:
        class FakeReasoningStreamingProvider:
            async def complete(self, request: ModelRequest):
                raise AssertionError("complete should not be called")

            async def stream(self, request: ModelRequest):
                del request
                yield ProviderStreamEvent.reasoning_delta("thinking ")
                yield ProviderStreamEvent.reasoning_delta("more")
                yield ProviderStreamEvent.text_delta("answer")
                yield ProviderStreamEvent.message_completed()
                yield ProviderStreamEvent.completed()

            async def capabilities(self):
                from runlet.core.models import ModelCapabilities

                return ModelCapabilities(
                    model_name="fake-reasoning",
                    context_window=4096,
                    supports_streaming=True,
                )

        observer = InMemoryObserver()
        runtime = Runtime(event_sink=observer)
        agent = Agent(name="assistant", instructions="Help.", model=FakeReasoningStreamingProvider())

        reasoning_deltas: list[str] = []
        async for event in runtime.stream(agent, "hi"):
            if event.type == "model.stream.reasoning_delta":
                reasoning_deltas.append(event.payload["delta"])

        self.assertEqual(reasoning_deltas, ["thinking ", "more"])
        completed_events = [event for event in observer.events if event.type == "run.completed"]
        self.assertEqual(len(completed_events), 1)
        self.assertEqual(completed_events[0].payload["output"], "answer")
        self.assertEqual(completed_events[0].payload["reasoning"], "thinking more")
