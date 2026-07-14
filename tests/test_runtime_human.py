import asyncio
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

    async def test_failed_approved_handler_leaves_checkpoint_resumable(self) -> None:
        attempts = 0

        async def refund(arguments, context):
            nonlocal attempts
            del arguments, context
            attempts += 1
            if attempts == 1:
                raise RuntimeError("temporary failure")
            return "refunded"

        store = InMemoryCheckpointStore()
        model = FakeModelProvider(
            [
                ModelResponse(
                    message=Message.assistant(""),
                    tool_calls=[ToolCall("call_1", "refund", {"order_id": "123", "amount": "10"})],
                ),
                ModelResponse(message=Message.assistant("done")),
            ]
        )
        runtime = Runtime(checkpoint_store=store)
        agent = approval_agent(model, refund)
        interrupted = await runtime.run(agent, "Refund it")
        response = HumanResponse(request_id=interrupted.interruption.id, action="approve")

        with self.assertRaisesRegex(RuntimeError, "temporary failure"):
            await runtime.resume(agent, checkpoint_id=interrupted.checkpoint_id, response=response)

        self.assertIsNotNone(await store.load(interrupted.checkpoint_id))
        result = await runtime.resume(agent, checkpoint_id=interrupted.checkpoint_id, response=response)
        self.assertEqual(result.output, "done")
        self.assertEqual(attempts, 2)

    async def test_resume_processes_remaining_normal_calls_in_order(self) -> None:
        calls: list[str] = []

        async def refund(arguments, context):
            del arguments, context
            calls.append("refund")
            return "refunded"

        async def lookup(arguments, context):
            del arguments, context
            calls.append("lookup")
            return "found"

        lookup_tool = ToolSpec(
            name="lookup",
            description="Look up an order.",
            input_schema={"type": "object", "required": [], "properties": {}},
            handler=lookup,
        )
        model = FakeModelProvider(
            [
                ModelResponse(
                    message=Message.assistant(""),
                    tool_calls=[ToolCall("call_1", "refund", {"order_id": "123", "amount": "10"}), ToolCall("call_2", "lookup")],
                ),
                ModelResponse(message=Message.assistant("done")),
            ]
        )
        runtime = Runtime()
        agent = Agent(
            name="support",
            instructions="Help.",
            model=model,
            tools=(approval_tool(refund), lookup_tool),
        )
        interrupted = await runtime.run(agent, "Refund it")

        result = await runtime.resume(
            agent,
            checkpoint_id=interrupted.checkpoint_id,
            response=HumanResponse(request_id=interrupted.interruption.id, action="approve"),
        )

        self.assertEqual(result.output, "done")
        self.assertEqual(calls, ["refund", "lookup"])
        continuation_messages = model.requests[1].messages
        self.assertEqual(continuation_messages[-3].role, "assistant")
        self.assertEqual(continuation_messages[-2].metadata["tool_call_id"], "call_1")
        self.assertEqual(continuation_messages[-1].metadata["tool_call_id"], "call_2")

    async def test_agent_name_mismatch_leaves_checkpoint_unconsumed(self) -> None:
        async def refund(arguments, context):
            del arguments, context
            return "refunded"

        store = InMemoryCheckpointStore()
        runtime = Runtime(checkpoint_store=store)
        model = FakeModelProvider(
            [ModelResponse(message=Message.assistant(""), tool_calls=[ToolCall("call_1", "refund", {"order_id": "123", "amount": "10"})])]
        )
        interrupted = await runtime.run(approval_agent(model, refund), "Refund it")
        wrong_agent = Agent(name="other", instructions="Help.", model=model, tools=(approval_tool(refund),))

        with self.assertRaisesRegex(ValueError, "agent"):
            await runtime.resume(
                wrong_agent,
                checkpoint_id=interrupted.checkpoint_id,
                response=HumanResponse(request_id=interrupted.interruption.id, action="approve"),
            )

        self.assertIsNotNone(await store.load(interrupted.checkpoint_id))

    async def test_invalid_choice_and_input_responses_leave_checkpoints_unconsumed(self) -> None:
        choice_model = FakeModelProvider(
            [
                ModelResponse(
                    message=Message.assistant(""),
                    tool_calls=[ToolCall("choice_1", "ask_human", {"kind": "choice", "prompt": "Pick", "options": [{"id": "a", "label": "A"}]})],
                )
            ]
        )
        input_model = FakeModelProvider(
            [ModelResponse(message=Message.assistant(""), tool_calls=[ToolCall("input_1", "ask_human", {"kind": "input", "prompt": "Reply"})])]
        )
        choice_store = InMemoryCheckpointStore()
        input_store = InMemoryCheckpointStore()
        choice_runtime = Runtime(checkpoint_store=choice_store)
        input_runtime = Runtime(checkpoint_store=input_store)
        choice_agent = Agent(name="support", instructions="Help.", model=choice_model, tools=(ask_human(),))
        input_agent = Agent(name="support", instructions="Help.", model=input_model, tools=(ask_human(),))
        choice = await choice_runtime.run(choice_agent, "Ask")
        input_request = await input_runtime.run(input_agent, "Ask")

        with self.assertRaisesRegex(ValueError, "listed"):
            await choice_runtime.resume(
                choice_agent,
                checkpoint_id=choice.checkpoint_id,
                response=HumanResponse(request_id=choice.interruption.id, action="select", value="missing"),
            )
        with self.assertRaisesRegex(ValueError, "submit"):
            await input_runtime.resume(
                input_agent,
                checkpoint_id=input_request.checkpoint_id,
                response=HumanResponse(request_id=input_request.interruption.id, action="select", value="text"),
            )

        self.assertIsNotNone(await choice_store.load(choice.checkpoint_id))
        self.assertIsNotNone(await input_store.load(input_request.checkpoint_id))

    async def test_cancelled_approved_handler_leaves_checkpoint_resumable(self) -> None:
        started = asyncio.Event()
        attempts = 0

        async def refund(arguments, context):
            nonlocal attempts
            del arguments, context
            attempts += 1
            if attempts == 1:
                started.set()
                await asyncio.Event().wait()
            return "refunded"

        store = InMemoryCheckpointStore()
        model = FakeModelProvider(
            [
                ModelResponse(
                    message=Message.assistant(""),
                    tool_calls=[ToolCall("call_1", "refund", {"order_id": "123", "amount": "10"})],
                ),
                ModelResponse(message=Message.assistant("done")),
            ]
        )
        runtime = Runtime(checkpoint_store=store)
        agent = approval_agent(model, refund)
        interrupted = await runtime.run(agent, "Refund it")
        response = HumanResponse(request_id=interrupted.interruption.id, action="approve")
        task = asyncio.create_task(runtime.resume(agent, checkpoint_id=interrupted.checkpoint_id, response=response))
        await started.wait()
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertIsNotNone(await store.load(interrupted.checkpoint_id))
        result = await runtime.resume(agent, checkpoint_id=interrupted.checkpoint_id, response=response)
        self.assertEqual(result.output, "done")

    async def test_resume_interrupts_again_for_remaining_gated_call(self) -> None:
        calls: list[str] = []

        async def refund(arguments, context):
            del context
            calls.append(arguments["order_id"])
            return "refunded"

        model = FakeModelProvider(
            [
                ModelResponse(
                    message=Message.assistant(""),
                    tool_calls=[
                        ToolCall("call_1", "refund", {"order_id": "first", "amount": "10"}),
                        ToolCall("call_2", "refund", {"order_id": "second", "amount": "20"}),
                    ],
                ),
                ModelResponse(message=Message.assistant("done")),
            ]
        )
        runtime = Runtime()
        agent = approval_agent(model, refund)
        first = await runtime.run(agent, "Refund both")

        second = await runtime.resume(
            agent,
            checkpoint_id=first.checkpoint_id,
            response=HumanResponse(request_id=first.interruption.id, action="approve"),
        )

        self.assertEqual(second.status, "interrupted")
        self.assertEqual(second.interruption.tool_call.id, "call_2")
        self.assertEqual(calls, ["first"])
        self.assertEqual(len(model.requests), 1)

        result = await runtime.resume(
            agent,
            checkpoint_id=second.checkpoint_id,
            response=HumanResponse(request_id=second.interruption.id, action="approve"),
        )

        self.assertEqual(result.output, "done")
        self.assertEqual(calls, ["first", "second"])
