import unittest

from runlet import Agent, Message, Runtime
from runlet.core import ToolCall
from runlet.core.events import InMemoryObserver
from runlet.core.models import ModelResponse
from runlet.integrations.tools import tool
from runlet.testing import FakeModelProvider


class RuntimeToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_executes_tool_then_returns_final_answer(self) -> None:
        @tool
        async def lookup(order_id: str) -> str:
            return f"order {order_id} shipped"

        model = FakeModelProvider(
            [
                ModelResponse(
                    message=Message.assistant(""),
                    tool_calls=[ToolCall(id="call_1", name="lookup", arguments={"order_id": "123"})],
                    final=False,
                ),
                ModelResponse(message=Message.assistant("order 123 shipped")),
            ]
        )
        observer = InMemoryObserver()
        runtime = Runtime(event_sink=observer)
        agent = Agent(name="support", instructions="Help.", model=model, tools=(lookup,))

        result = await runtime.run(agent, "where is 123?")

        self.assertEqual(result.output, "order 123 shipped")
        self.assertIn("tool.started", [event.type for event in observer.events])
        self.assertIn("tool.completed", [event.type for event in observer.events])
        self.assertEqual(len(model.requests), 2)
        self.assertEqual(model.requests[1].messages[-1].role, "tool")
