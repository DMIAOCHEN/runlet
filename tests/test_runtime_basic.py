import unittest

from runlet import Agent, Message, Runtime
from runlet.core.events import InMemoryObserver
from runlet.core.models import ModelResponse
from runlet.testing import FakeModelProvider


class RuntimeBasicTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_returns_final_answer(self) -> None:
        model = FakeModelProvider([ModelResponse(message=Message.assistant("hello"))])
        observer = InMemoryObserver()
        runtime = Runtime(event_sink=observer)
        agent = Agent(name="assistant", instructions="Be helpful.", model=model)

        result = await runtime.run(agent, "hi")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.output, "hello")
        self.assertEqual(
            [event.type for event in observer.events],
            ["run.started", "context.budget_checked", "model.requested", "model.completed", "run.completed"],
        )
        self.assertEqual(model.requests[0].messages[0].role, "system")
        self.assertEqual(model.requests[0].messages[1].text, "hi")
