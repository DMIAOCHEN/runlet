import unittest

from runlet import Agent, HumanResponse, Message, Runtime, ToolCall
from runlet.core.events import CompositeEventSink, InMemoryObserver, RuntimeEvent
from runlet.core.models import ModelResponse
from runlet.integrations.human import ask_human
from runlet.testing import FakeModelProvider


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

    async def test_human_events_do_not_include_request_or_response_content(self) -> None:
        model = FakeModelProvider(
            [
                ModelResponse(
                    message=Message.assistant(""),
                    tool_calls=[
                        ToolCall(
                            "call_1",
                            "ask_human",
                            {
                                "kind": "input",
                                "prompt": "Enter your private account number.",
                            },
                        )
                    ],
                ),
                ModelResponse(message=Message.assistant("Thanks.")),
            ]
        )
        observer = InMemoryObserver()
        runtime = Runtime(event_sink=observer)
        agent = Agent(name="support", instructions="Help.", model=model, tools=(ask_human(),))

        interrupted = await runtime.run(agent, "Ask for my account number")
        await runtime.resume(
            agent,
            checkpoint_id=interrupted.checkpoint_id,
            response=HumanResponse(
                request_id=interrupted.interruption.id,
                action="submit",
                value="account-12345",
            ),
        )

        events = {event.type: event for event in observer.events}
        expected = {
            "request_id": interrupted.interruption.id,
            "checkpoint_id": interrupted.checkpoint_id,
            "kind": "input",
        }
        self.assertEqual(events["human.requested"].payload, expected)
        self.assertEqual(
            events["human.responded"].payload,
            {**expected, "action": "submit"},
        )

    async def test_rejected_human_response_event_omits_submitted_value(self) -> None:
        model = FakeModelProvider(
            [
                ModelResponse(
                    message=Message.assistant(""),
                    tool_calls=[
                        ToolCall(
                            "call_1",
                            "ask_human",
                            {
                                "kind": "choice",
                                "prompt": "Choose a private plan.",
                                "options": [{"id": "pro", "label": "Pro"}],
                            },
                        )
                    ],
                )
            ]
        )
        observer = InMemoryObserver()
        runtime = Runtime(event_sink=observer)
        agent = Agent(name="support", instructions="Help.", model=model, tools=(ask_human(),))

        interrupted = await runtime.run(agent, "Ask for my plan")

        with self.assertRaisesRegex(ValueError, "listed"):
            await runtime.resume(
                agent,
                checkpoint_id=interrupted.checkpoint_id,
                response=HumanResponse(
                    request_id=interrupted.interruption.id,
                    action="select",
                    value="private-plan-value",
                ),
            )

        event = observer.events[-1]
        self.assertEqual(event.type, "human.response_rejected")
        self.assertEqual(
            event.payload,
            {
                "request_id": interrupted.interruption.id,
                "checkpoint_id": interrupted.checkpoint_id,
                "kind": "choice",
                "action": "select",
            },
        )

    async def test_stale_human_response_emits_rejection_event(self) -> None:
        model = FakeModelProvider(
            [
                ModelResponse(
                    message=Message.assistant(""),
                    tool_calls=[
                        ToolCall("call_1", "ask_human", {"kind": "input", "prompt": "Enter a secret."})
                    ],
                )
            ]
        )
        observer = InMemoryObserver()
        runtime = Runtime(event_sink=observer)
        agent = Agent(name="support", instructions="Help.", model=model, tools=(ask_human(),))
        interrupted = await runtime.run(agent, "Ask for a secret")

        with self.assertRaisesRegex(ValueError, "request"):
            await runtime.resume(
                agent,
                checkpoint_id=interrupted.checkpoint_id,
                response=HumanResponse(request_id="stale-id", action="submit", value="private-secret"),
            )

        event = observer.events[-1]
        self.assertEqual(event.type, "human.response_rejected")
        self.assertEqual(
            event.payload,
            {
                "request_id": interrupted.interruption.id,
                "checkpoint_id": interrupted.checkpoint_id,
                "kind": "input",
                "action": "submit",
            },
        )
