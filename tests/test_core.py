import unittest

from runlet import Agent, Message, RunResult, ToolCall, Usage
from runlet.errors import ContextOverflowError, RunletError


class CoreObjectTests(unittest.TestCase):
    def test_message_text_constructor(self) -> None:
        message = Message.user("hello")

        self.assertEqual(message.role, "user")
        self.assertEqual(message.text, "hello")
        self.assertEqual(message.metadata, {})

    def test_agent_model_must_be_instance(self) -> None:
        model = object()
        agent = Agent(name="assistant", instructions="help", model=model)

        self.assertIs(agent.model, model)
        self.assertEqual(agent.tools, ())
        self.assertEqual(agent.hooks, ())

    def test_run_result_completed(self) -> None:
        result = RunResult.completed(run_id="run_1", output="done", usage=Usage(input_tokens=1, output_tokens=2))

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.output, "done")
        self.assertEqual(result.usage.total_tokens, 3)

    def test_tool_call_has_arguments(self) -> None:
        call = ToolCall(id="call_1", name="lookup", arguments={"id": "123"})

        self.assertEqual(call.arguments["id"], "123")

    def test_errors_carry_code_and_message(self) -> None:
        error = ContextOverflowError("too large")

        self.assertIsInstance(error, RunletError)
        self.assertEqual(error.code, "context_overflow")
        self.assertEqual(str(error), "too large")
