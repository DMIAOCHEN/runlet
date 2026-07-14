import json
import unittest

from runlet import Agent, HumanResponse, Message, Runtime, ToolCall
from runlet.core.events import InMemoryObserver
from runlet.core.models import ModelResponse
from runlet.integrations.human import ask_human
from runlet.integrations.tools import ToolSpec
from runlet.runtime.checkpoints import InMemoryCheckpointStore
from runlet.testing import FakeModelProvider


def approval_tool(handler) -> ToolSpec:
    return ToolSpec(
        name="refund",
        description="Refund an order.",
        input_schema={
            "type": "object",
            "required": ["order_id", "amount"],
            "properties": {"order_id": {"type": "string"}, "amount": {"type": "string"}},
        },
        handler=handler,
        requires_approval=True,
    )


def approval_agent(model, handler) -> Agent:
    return Agent(name="support", instructions="Help.", model=model, tools=(approval_tool(handler),))


class RuntimeHumanTests(unittest.IsolatedAsyncioTestCase):
    async def test_approval_gate_does_not_execute_before_resume(self) -> None:
        calls: list[dict[str, str]] = []

        async def refund(arguments, context):
            del context
            calls.append(arguments)
            return "refunded"

        model = FakeModelProvider(
            [
                ModelResponse(
                    message=Message.assistant(""),
                    tool_calls=[ToolCall("call_1", "refund", {"order_id": "123", "amount": "10"})],
                    final=False,
                ),
                ModelResponse(message=Message.assistant("done")),
            ]
        )
        observer = InMemoryObserver()
        runtime = Runtime(event_sink=observer)

        result = await runtime.run(approval_agent(model, refund), "Refund it")

        self.assertEqual(result.status, "interrupted")
        self.assertEqual(calls, [])
        self.assertEqual(result.interruption.kind, "tool_approval")
        self.assertIsNotNone(result.checkpoint_id)
        self.assertNotIn("tool.started", [event.type for event in observer.events])

    async def test_approval_resume_executes_edited_arguments(self) -> None:
        calls: list[dict[str, str]] = []

        async def refund(arguments, context):
            del context
            calls.append(arguments)
            return "refunded"

        model = FakeModelProvider(
            [
                ModelResponse(
                    message=Message.assistant(""),
                    tool_calls=[ToolCall("call_1", "refund", {"order_id": "123", "amount": "10"})],
                    final=False,
                ),
                ModelResponse(message=Message.assistant("done")),
            ]
        )
        runtime = Runtime()
        agent = approval_agent(model, refund)
        interrupted = await runtime.run(agent, "Refund it")

        result = await runtime.resume(
            agent,
            checkpoint_id=interrupted.checkpoint_id,
            response=HumanResponse(
                request_id=interrupted.interruption.id,
                action="approve",
                edited_arguments={"order_id": "123", "amount": "8"},
            ),
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.run_id, interrupted.run_id)
        self.assertEqual(calls, [{"order_id": "123", "amount": "8"}])

    async def test_rejection_adds_tool_message_for_original_call(self) -> None:
        async def refund(arguments, context):
            raise AssertionError("rejected tool must not execute")

        model = FakeModelProvider(
            [
                ModelResponse(
                    message=Message.assistant(""),
                    tool_calls=[ToolCall("call_1", "refund", {"order_id": "123", "amount": "10"})],
                    final=False,
                ),
                ModelResponse(message=Message.assistant("request declined")),
            ]
        )
        runtime = Runtime()
        agent = approval_agent(model, refund)
        interrupted = await runtime.run(agent, "Refund it")

        result = await runtime.resume(
            agent,
            checkpoint_id=interrupted.checkpoint_id,
            response=HumanResponse(
                request_id=interrupted.interruption.id,
                action="reject",
                metadata={"reason": "outside policy"},
            ),
        )

        self.assertEqual(result.output, "request declined")
        tool_message = model.requests[1].messages[-1]
        self.assertEqual(tool_message.text, "Human rejected tool call.")
        self.assertEqual(
            tool_message.metadata,
            {
                "tool_call_id": "call_1",
                "name": "refund",
                "human_rejected": True,
                "reason": "outside policy",
            },
        )

    async def test_stale_request_id_leaves_checkpoint_unconsumed(self) -> None:
        async def refund(arguments, context):
            del arguments, context
            return "refunded"

        store = InMemoryCheckpointStore()
        model = FakeModelProvider(
            [ModelResponse(message=Message.assistant(""), tool_calls=[ToolCall("call_1", "refund", {"order_id": "123", "amount": "10"})])]
        )
        runtime = Runtime(checkpoint_store=store)
        agent = approval_agent(model, refund)
        interrupted = await runtime.run(agent, "Refund it")

        with self.assertRaisesRegex(ValueError, "request"):
            await runtime.resume(
                agent,
                checkpoint_id=interrupted.checkpoint_id,
                response=HumanResponse(request_id="stale", action="approve"),
            )

        self.assertIsNotNone(await store.load(interrupted.checkpoint_id))
        self.assertEqual(len(model.requests), 1)

    async def test_invalid_edited_arguments_leave_checkpoint_unconsumed(self) -> None:
        async def refund(arguments, context):
            del arguments, context
            return "refunded"

        store = InMemoryCheckpointStore()
        model = FakeModelProvider(
            [ModelResponse(message=Message.assistant(""), tool_calls=[ToolCall("call_1", "refund", {"order_id": "123", "amount": "10"})])]
        )
        runtime = Runtime(checkpoint_store=store)
        agent = approval_agent(model, refund)
        interrupted = await runtime.run(agent, "Refund it")

        with self.assertRaisesRegex(ValueError, "amount"):
            await runtime.resume(
                agent,
                checkpoint_id=interrupted.checkpoint_id,
                response=HumanResponse(
                    request_id=interrupted.interruption.id,
                    action="approve",
                    edited_arguments={"order_id": "123"},
                ),
            )

        self.assertIsNotNone(await store.load(interrupted.checkpoint_id))
        self.assertEqual(len(model.requests), 1)

    async def test_human_input_resume_returns_tool_result_with_original_call_id(self) -> None:
        model = FakeModelProvider(
            [
                ModelResponse(
                    message=Message.assistant(""),
                    tool_calls=[
                        ToolCall(
                            "call_1",
                            "ask_human",
                            {"kind": "choice", "prompt": "Pick", "options": [{"id": "a", "label": "A"}]},
                        )
                    ],
                    final=False,
                ),
                ModelResponse(message=Message.assistant("selected")),
            ]
        )
        runtime = Runtime()
        agent = Agent(name="support", instructions="Help.", model=model, tools=(ask_human(),))
        interrupted = await runtime.run(agent, "Ask me")

        result = await runtime.resume(
            agent,
            checkpoint_id=interrupted.checkpoint_id,
            response=HumanResponse(request_id=interrupted.interruption.id, action="select", value="a"),
        )

        self.assertEqual(result.output, "selected")
        tool_message = model.requests[1].messages[-1]
        self.assertEqual(json.loads(tool_message.text), {"kind": "choice", "value": "a"})
        self.assertEqual(
            tool_message.metadata,
            {"tool_call_id": "call_1", "name": "ask_human", "human_input": True},
        )
