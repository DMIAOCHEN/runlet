import unittest
from collections.abc import AsyncIterator

from runlet import Agent, Runtime
from runlet.core.events import InMemoryObserver
from runlet.core.models import ModelCapabilities, ModelRequest, ModelResponse, ProviderStreamEvent
from runlet.integrations.tools import tool


class FakeToolStreamingProvider:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self._round = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError("complete should not be called in streaming tool loop test")

    async def stream(self, request: ModelRequest) -> AsyncIterator[ProviderStreamEvent]:
        self.requests.append(request)
        self._round += 1
        if self._round == 1:
            yield ProviderStreamEvent.text_delta("Checking order ")
            yield ProviderStreamEvent.tool_call_delta(
                call_id="call_1",
                name="lookup_order",
                arguments_delta='{"order_id":"123"}',
            )
            yield ProviderStreamEvent.tool_call_completed(
                call_id="call_1",
                name="lookup_order",
                arguments={"order_id": "123"},
            )
            yield ProviderStreamEvent.completed()
            return

        if self._round == 2:
            yield ProviderStreamEvent.text_delta("Shipped via warehouse ")
            yield ProviderStreamEvent.tool_call_completed(
                call_id="call_2",
                name="lookup_warehouse",
                arguments={"warehouse_id": "w1"},
            )
            yield ProviderStreamEvent.completed()
            return

        yield ProviderStreamEvent.text_delta("A and arriving tomorrow.")
        yield ProviderStreamEvent.message_completed()
        yield ProviderStreamEvent.completed()

    async def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_name="fake-tool-stream",
            context_window=4096,
            supports_streaming=True,
        )


class RuntimeStreamingToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_executes_tools_and_continues_streaming_across_rounds(self) -> None:
        @tool
        async def lookup_order(order_id: str) -> str:
            return f"warehouse:w1 for {order_id}"

        @tool
        async def lookup_warehouse(warehouse_id: str) -> str:
            return f"{warehouse_id}:A"

        model = FakeToolStreamingProvider()
        observer = InMemoryObserver()
        runtime = Runtime(event_sink=observer)
        agent = Agent(
            name="assistant",
            instructions="Help users with orders.",
            model=model,
            tools=(lookup_order, lookup_warehouse),
        )

        deltas: list[str] = []
        async for event in runtime.stream(agent, "Where is order 123?"):
            if event.type == "model.stream.delta":
                deltas.append(event.payload["delta"])

        self.assertEqual(
            deltas,
            ["Checking order ", "Shipped via warehouse ", "A and arriving tomorrow."],
        )
        self.assertEqual(len(model.requests), 3)
        self.assertEqual(model.requests[1].messages[-2].role, "assistant")
        self.assertEqual(
            model.requests[1].messages[-2].metadata["tool_calls"],
            [{"id": "call_1", "name": "lookup_order", "arguments": {"order_id": "123"}}],
        )
        self.assertEqual(model.requests[1].messages[-1].role, "tool")
        self.assertEqual(model.requests[2].messages[-2].role, "assistant")
        self.assertEqual(
            model.requests[2].messages[-2].metadata["tool_calls"],
            [{"id": "call_2", "name": "lookup_warehouse", "arguments": {"warehouse_id": "w1"}}],
        )
        self.assertEqual(model.requests[2].messages[-1].role, "tool")
        self.assertIn("tool.started", [event.type for event in observer.events])
        self.assertIn("tool.completed", [event.type for event in observer.events])
        self.assertEqual(observer.events[-1].type, "run.completed")
