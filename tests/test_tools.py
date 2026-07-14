import unittest
from typing import Any

from runlet.core import ToolCall
from runlet.integrations import ToolContext, ToolSpec, execute_tool_call, tool


class ToolTests(unittest.IsolatedAsyncioTestCase):
    def test_tool_spec_defaults_to_not_requiring_approval(self) -> None:
        async def handler(arguments: dict[str, Any], context: ToolContext) -> str:
            return "unused"

        spec = ToolSpec(name="read", description="", input_schema={}, handler=handler)

        self.assertFalse(spec.requires_approval)

    async def test_decorator_creates_tool_spec(self) -> None:
        @tool
        async def lookup(order_id: str) -> str:
            return f"order:{order_id}"

        self.assertEqual(lookup.name, "lookup")
        self.assertIn("order_id", lookup.input_schema["properties"])

    async def test_execute_tool_call_returns_tool_result(self) -> None:
        async def handler(arguments: dict[str, Any], context: ToolContext) -> str:
            return f"hello {arguments['name']}"

        spec = ToolSpec(
            name="greet",
            description="Greet a user.",
            input_schema={"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}},
            handler=handler,
        )

        result = await execute_tool_call(
            ToolCall(id="call_1", name="greet", arguments={"name": "Ada"}),
            {"greet": spec},
            ToolContext(run_id="run_1"),
        )

        self.assertEqual(result.call_id, "call_1")
        self.assertEqual(result.content, "hello Ada")

    async def test_execute_tool_call_rejects_missing_required_argument(self) -> None:
        async def handler(arguments: dict[str, Any], context: ToolContext) -> str:
            return "never"

        spec = ToolSpec(
            name="greet",
            description="Greet a user.",
            input_schema={"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}},
            handler=handler,
        )

        with self.assertRaises(ValueError):
            await execute_tool_call(
                ToolCall(id="call_1", name="greet", arguments={}),
                {"greet": spec},
                ToolContext(run_id="run_1"),
            )
