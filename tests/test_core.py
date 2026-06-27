import unittest

from runlet import Agent, Message, RunContext, RunResult, ToolCall, ToolResult, Usage
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
        self.assertEqual(call.metadata, {})

    def test_tool_result_has_content_and_metadata(self) -> None:
        result = ToolResult(call_id="call_1", name="lookup", content="found")

        self.assertEqual(result.content, "found")
        self.assertEqual(result.metadata, {})

    def test_usage_defaults_to_actual_source(self) -> None:
        self.assertEqual(Usage().source, "actual")

    def test_run_context_has_mutable_messages_and_usage(self) -> None:
        agent = Agent(name="assistant", instructions="help", model=object())
        context = RunContext(run_id="run_1", agent=agent)

        context.messages.append(Message.user("hello"))

        self.assertEqual(context.messages[0].text, "hello")
        self.assertEqual(context.usage.total_tokens, 0)

    def test_run_result_failed_has_string_error_and_messages(self) -> None:
        result = RunResult.failed(run_id="run_1", error="boom")

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error, "boom")
        self.assertIsInstance(result.error, str)
        self.assertEqual(result.messages, ())

    def test_errors_carry_code_and_message(self) -> None:
        error = ContextOverflowError("too large")

        self.assertIsInstance(error, RunletError)
        self.assertEqual(error.code, "context_overflow")
        self.assertEqual(str(error), "too large")

    def test_errors_are_exported_from_package_root(self) -> None:
        from runlet import ContextOverflowError as RootContextOverflowError
        from runlet import RunletError as RootRunletError

        self.assertEqual(RootRunletError.code, "runlet_error")
        self.assertEqual(RootContextOverflowError.code, "context_overflow")
