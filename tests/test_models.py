import unittest

from runlet.core import Message, ToolCall, Usage
from runlet.core.models import ModelCapabilities, ModelRequest, ModelResponse
from runlet.testing import FakeModelProvider, FakeStreamingModelProvider


class ModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_fake_model_returns_response(self) -> None:
        provider = FakeModelProvider([ModelResponse(message=Message.assistant("hello"))])
        request = ModelRequest(messages=[Message.user("hi")])

        response = await provider.complete(request)

        self.assertEqual(response.message.text, "hello")

    async def test_fake_model_returns_capabilities(self) -> None:
        provider = FakeModelProvider([], capabilities=ModelCapabilities(model_name="fake", context_window=100))

        capabilities = await provider.capabilities()

        self.assertEqual(capabilities.model_name, "fake")
        self.assertEqual(capabilities.context_window, 100)

    async def test_fake_streaming_model_yields_events(self) -> None:
        provider = FakeStreamingModelProvider(["hel", "lo"])
        request = ModelRequest(messages=[Message.user("hi")])

        chunks: list[str] = []
        async for event in provider.stream(request):
            chunks.append(event.delta)

        self.assertEqual(chunks, ["hel", "lo", ""])

    def test_model_response_can_include_tool_calls_and_usage(self) -> None:
        response = ModelResponse(
            message=Message.assistant(""),
            tool_calls=[ToolCall(id="call_1", name="lookup", arguments={"id": "1"})],
            usage=Usage(input_tokens=5, output_tokens=3),
        )

        self.assertEqual(response.tool_calls[0].name, "lookup")
        self.assertEqual(response.usage.total_tokens, 8)
