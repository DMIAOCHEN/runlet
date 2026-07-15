# pyright: strict, reportArgumentType=false, reportMissingParameterType=false, reportOptionalMemberAccess=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownVariableType=false

import unittest
from collections.abc import AsyncIterator

from runlet import Agent, HumanResponse, Runtime
from runlet.core.events import InMemoryObserver
from runlet.core.models import ModelCapabilities, ModelRequest, ModelResponse, ProviderStreamEvent
from runlet.core.messages import Message
from runlet.integrations.human import ask_human
from runlet.integrations.tools import ToolSpec


class StreamingHumanProvider:
    def __init__(self, tool_name: str, arguments: dict[str, object]) -> None:
        self.tool_name = tool_name
        self.arguments = arguments
        self.stream_requests: list[ModelRequest] = []
        self.complete_requests: list[ModelRequest] = []
        self.stream_closed = False

    async def stream(self, request: ModelRequest) -> AsyncIterator[ProviderStreamEvent]:
        self.stream_requests.append(request)
        try:
            yield ProviderStreamEvent.text_delta("Checking that now. ")
            yield ProviderStreamEvent.tool_call_completed("call_1", self.tool_name, self.arguments)
            yield ProviderStreamEvent.message_completed()
        finally:
            self.stream_closed = True

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.complete_requests.append(request)
        return ModelResponse(message=Message.assistant("Completed after human input."))

    async def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(model_name="streaming-human", context_window=4096, supports_streaming=True)


class MultiToolStreamingHumanProvider:
    def __init__(self, calls: list[tuple[str, str, dict[str, object]]]) -> None:
        self.calls = calls
        self.stream_requests: list[ModelRequest] = []
        self.complete_requests: list[ModelRequest] = []
        self.stream_closed = False

    async def stream(self, request: ModelRequest) -> AsyncIterator[ProviderStreamEvent]:
        self.stream_requests.append(request)
        try:
            yield ProviderStreamEvent.text_delta("Checking that now. ")
            for call_id, name, arguments in self.calls:
                yield ProviderStreamEvent.tool_call_completed(call_id, name, arguments)
            yield ProviderStreamEvent.message_completed()
        finally:
            self.stream_closed = True

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.complete_requests.append(request)
        return ModelResponse(message=Message.assistant("Completed after human input."))

    async def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(model_name="streaming-human", context_window=4096, supports_streaming=True)

class RuntimeStreamingHumanTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_yields_text_then_approval_interruption_and_resume_executes_tool(self) -> None:
        executed_calls: list[dict[str, str]] = []

        async def refund(arguments, context) -> str:
            del context
            order_id = arguments["order_id"]
            executed_calls.append({"order_id": order_id})
            return "refunded"

        refund_tool = ToolSpec(
            name="refund",
            description="Refund an order.",
            input_schema={
                "type": "object",
                "required": ["order_id"],
                "properties": {"order_id": {"type": "string"}},
            },
            handler=refund,
            requires_approval=True,
        )

        model = StreamingHumanProvider("refund", {"order_id": "123"})
        observer = InMemoryObserver()
        runtime = Runtime(event_sink=observer)
        agent = Agent(name="support", instructions="Help.", model=model, tools=(refund_tool,))

        events = [event async for event in runtime.stream(agent, "Refund it")]

        self.assertEqual(
            [event.type for event in events],
            ["model.stream.delta", "human.requested", "run.interrupted"],
        )
        self.assertEqual(executed_calls, [])
        interrupted = events[-1]
        self.assertEqual(interrupted.payload["kind"], "tool_approval")
        self.assertEqual(len(model.stream_requests), 1)
        self.assertNotIn("tool.started", [event.type for event in observer.events])
        self.assertEqual(
            [event.type for event in observer.events if event.type in {"human.requested", "run.interrupted"}],
            ["human.requested", "run.interrupted"],
        )
        self.assertTrue(model.stream_closed)

        result = await runtime.resume(
            agent,
            checkpoint_id=interrupted.payload["checkpoint_id"],
            response=HumanResponse(request_id=events[-2].payload["request_id"], action="approve"),
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(executed_calls, [{"order_id": "123"}])
        self.assertEqual(model.complete_requests[-1].messages[-1].text, "refunded")

    async def test_stream_yields_choice_interruption_and_resume_adds_human_result(self) -> None:
        model = StreamingHumanProvider(
            "ask_human",
            {
                "kind": "choice",
                "prompt": "Which refund method?",
                "options": [{"id": "credit", "label": "Store credit"}],
            },
        )
        runtime = Runtime()
        agent = Agent(name="support", instructions="Help.", model=model, tools=(ask_human(),))

        events = [event async for event in runtime.stream(agent, "Refund it")]

        self.assertEqual(
            [event.type for event in events],
            ["model.stream.delta", "human.requested", "run.interrupted"],
        )
        self.assertEqual(events[-2].payload["kind"], "choice")
        self.assertEqual(events[-1].payload["kind"], "choice")

        result = await runtime.resume(
            agent,
            checkpoint_id=events[-1].payload["checkpoint_id"],
            response=HumanResponse(request_id=events[-2].payload["request_id"], action="select", value="credit"),
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(model.complete_requests[-1].messages[-1].metadata["human_input"], True)

    async def test_stream_preserves_normal_call_after_gated_call_until_resume(self) -> None:
        executed: list[str] = []

        async def refund(arguments, context) -> str:
            del arguments, context
            executed.append("refund")
            return "refunded"

        async def lookup(arguments, context) -> str:
            del arguments, context
            executed.append("lookup")
            return "found"

        model = MultiToolStreamingHumanProvider(
            [
                ("call_1", "refund", {"order_id": "123"}),
                ("call_2", "lookup", {}),
            ]
        )
        runtime = Runtime()
        agent = Agent(
            name="support",
            instructions="Help.",
            model=model,
            tools=(
                ToolSpec("refund", "Refund.", {"type": "object", "required": ["order_id"]}, refund, requires_approval=True),
                ToolSpec("lookup", "Look up.", {"type": "object"}, lookup),
            ),
        )

        events = [event async for event in runtime.stream(agent, "Refund it")]
        self.assertEqual([event.type for event in events], ["model.stream.delta", "human.requested", "run.interrupted"])
        self.assertTrue(model.stream_closed)
        self.assertEqual(executed, [])

        result = await runtime.resume(
            agent,
            checkpoint_id=events[-1].payload["checkpoint_id"],
            response=HumanResponse(request_id=events[-2].payload["request_id"], action="approve"),
        )

        self.assertEqual(result.output, "Completed after human input.")
        self.assertEqual(executed, ["refund", "lookup"])
        messages = model.complete_requests[-1].messages
        self.assertEqual(messages[-3].metadata["tool_calls"], [
            {"id": "call_1", "name": "refund", "arguments": {"order_id": "123"}},
            {"id": "call_2", "name": "lookup", "arguments": {}},
        ])
        self.assertEqual([message.metadata["tool_call_id"] for message in messages[-2:]], ["call_1", "call_2"])

    async def test_stream_two_gated_calls_interrupt_and_resume_in_order(self) -> None:
        executed: list[str] = []

        async def refund(arguments, context) -> str:
            del context
            executed.append(arguments["order_id"])
            return "refunded"

        model = MultiToolStreamingHumanProvider(
            [
                ("call_1", "refund", {"order_id": "first"}),
                ("call_2", "refund", {"order_id": "second"}),
            ]
        )
        observer = InMemoryObserver()
        runtime = Runtime(event_sink=observer)
        agent = Agent(
            name="support",
            instructions="Help.",
            model=model,
            tools=(ToolSpec("refund", "Refund.", {"type": "object", "required": ["order_id"]}, refund, requires_approval=True),),
        )

        events = [event async for event in runtime.stream(agent, "Refund both")]
        second = await runtime.resume(
            agent,
            checkpoint_id=events[-1].payload["checkpoint_id"],
            response=HumanResponse(request_id=events[-2].payload["request_id"], action="approve"),
        )

        self.assertEqual(second.status, "interrupted")
        self.assertEqual(second.interruption.tool_call.id, "call_2")
        self.assertEqual(executed, ["first"])

        result = await runtime.resume(
            agent,
            checkpoint_id=second.checkpoint_id,
            response=HumanResponse(request_id=second.interruption.id, action="approve"),
        )

        self.assertEqual(result.output, "Completed after human input.")
        self.assertEqual(executed, ["first", "second"])
        self.assertEqual(
            [event.type for event in observer.events if event.type in {"human.requested", "run.interrupted"}],
            ["human.requested", "run.interrupted", "human.requested", "run.interrupted"],
        )
