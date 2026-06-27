import unittest

from runlet import Agent, Message, Runtime
from runlet.core import ToolCall
from runlet.core.events import InMemoryObserver
from runlet.core.models import ModelRequest, ModelResponse
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
                    reasoning="first step ",
                ),
                ModelResponse(message=Message.assistant("order 123 shipped"), reasoning="second step"),
            ]
        )
        observer = InMemoryObserver()
        runtime = Runtime(event_sink=observer)
        agent = Agent(name="support", instructions="Help.", model=model, tools=(lookup,))

        result = await runtime.run(agent, "where is 123?")

        self.assertEqual(result.output, "order 123 shipped")
        self.assertEqual(result.reasoning, "first step second step")
        self.assertIn("tool.started", [event.type for event in observer.events])
        self.assertIn("tool.completed", [event.type for event in observer.events])
        self.assertEqual(len(model.requests), 2)
        self.assertEqual(model.requests[1].messages[-2].role, "assistant")
        self.assertEqual(
            model.requests[1].messages[-2].metadata["tool_calls"],
            [{"id": "call_1", "name": "lookup", "arguments": {"order_id": "123"}}],
        )
        self.assertEqual(model.requests[1].messages[-1].role, "tool")

    async def test_runtime_run_request_preserves_request_options(self) -> None:
        model = FakeModelProvider([ModelResponse(message=Message.assistant("done"))])
        runtime = Runtime()
        agent = Agent(name="support", instructions="Help.", model=model)
        request = ModelRequest(
            messages=[Message.system("Override system."), Message.user("hi")],
            options={"openai": {"extra_body": {"store": False}}},
        )

        result = await runtime.run_request(agent, request)

        self.assertEqual(result.output, "done")
        self.assertEqual(result.reasoning, "")
        self.assertEqual(
            model.requests[0].options,
            {"openai": {"extra_body": {"store": False}}},
        )
        self.assertEqual(model.requests[0].messages[0].text, "Override system.")
