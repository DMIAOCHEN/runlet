import sys
import types
import unittest
from unittest.mock import patch

from runlet.core import Message
from runlet.core.models import ModelRequest
from runlet.core.runs import Usage


class FakeOpenAIResponse:
    def __init__(self, output_text: str, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.output_text = output_text
        self.usage = types.SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)


class RecordingResponsesAPI:
    def __init__(self, response: FakeOpenAIResponse) -> None:
        self._response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> FakeOpenAIResponse:
        self.calls.append(kwargs)
        return self._response


class RecordingClient:
    def __init__(self, response: FakeOpenAIResponse) -> None:
        self.responses = RecordingResponsesAPI(response)


class OpenAIProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_maps_messages_and_returns_model_response(self) -> None:
        from runlet.providers.openai import OpenAIResponsesProvider

        fake_response = FakeOpenAIResponse("hello back", input_tokens=5, output_tokens=7)
        client = RecordingClient(fake_response)
        provider = OpenAIResponsesProvider(model="gpt-test", client=client)
        request = ModelRequest(
            messages=[
                Message.system("Be helpful."),
                Message.user("Hi"),
                Message.assistant("Previous reply"),
            ]
        )

        response = await provider.complete(request)

        self.assertEqual(response.message.role, "assistant")
        self.assertEqual(response.message.text, "hello back")
        self.assertEqual(response.usage, Usage(input_tokens=5, output_tokens=7))
        self.assertEqual(client.responses.calls[0]["model"], "gpt-test")
        self.assertEqual(
            client.responses.calls[0]["input"],
            [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Previous reply"},
            ],
        )

    async def test_capabilities_are_conservative(self) -> None:
        from runlet.providers.openai import OpenAIResponsesProvider

        provider = OpenAIResponsesProvider(model="gpt-test", client=RecordingClient(FakeOpenAIResponse("ok")))

        capabilities = await provider.capabilities()

        self.assertEqual(capabilities.model_name, "gpt-test")
        self.assertFalse(capabilities.supports_tools)
        self.assertFalse(capabilities.supports_parallel_tool_calls)
        self.assertFalse(capabilities.supports_streaming)
        self.assertGreater(capabilities.context_window, 0)

    async def test_tool_messages_are_rejected(self) -> None:
        from runlet.providers.openai import OpenAIResponsesProvider

        provider = OpenAIResponsesProvider(model="gpt-test", client=RecordingClient(FakeOpenAIResponse("ok")))
        request = ModelRequest(messages=[Message.tool("tool output")])

        with self.assertRaises(ValueError):
            await provider.complete(request)

    async def test_stream_is_not_implemented(self) -> None:
        from runlet.providers.openai import OpenAIResponsesProvider

        provider = OpenAIResponsesProvider(model="gpt-test", client=RecordingClient(FakeOpenAIResponse("ok")))

        with self.assertRaises(NotImplementedError):
            async for _ in provider.stream(ModelRequest(messages=[Message.user("hi")])):
                self.fail("stream should not yield")

    def test_missing_openai_dependency_raises_helpful_error(self) -> None:
        sys.modules.pop("openai", None)

        with patch.dict(sys.modules, {"openai": None}):
            from runlet.providers.openai import OpenAIResponsesProvider

            with self.assertRaisesRegex(RuntimeError, "pip install .*openai"):
                OpenAIResponsesProvider(model="gpt-test")
