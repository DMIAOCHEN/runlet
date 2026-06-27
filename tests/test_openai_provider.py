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
        self.stream_events: list[object] = []

    def create(self, **kwargs: object) -> FakeOpenAIResponse:
        self.calls.append(kwargs)
        return self._response

    def stream(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.stream_events


class RecordingResponseStreamManager:
    def __init__(self, events: list[object]) -> None:
        self._events = events
        self.entered = False
        self.exited = False

    def __enter__(self) -> list[object]:
        self.entered = True
        return self._events

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.exited = True


class ContextManagedResponsesAPI(RecordingResponsesAPI):
    def __init__(self, response: FakeOpenAIResponse) -> None:
        super().__init__(response)
        self.last_manager: RecordingResponseStreamManager | None = None

    def stream(self, **kwargs: object) -> RecordingResponseStreamManager:
        self.calls.append(kwargs)
        manager = RecordingResponseStreamManager(self.stream_events)
        self.last_manager = manager
        return manager


class ContextManagedClient:
    def __init__(self, response: FakeOpenAIResponse) -> None:
        self.responses = ContextManagedResponsesAPI(response)


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

    async def test_provider_forwards_openai_extra_body_options(self) -> None:
        from runlet.providers.openai import OpenAIResponsesProvider

        fake_response = FakeOpenAIResponse("hello back")
        client = RecordingClient(fake_response)
        provider = OpenAIResponsesProvider(model="gpt-test", client=client)
        request = ModelRequest(
            messages=[Message.user("Hi")],
            options={"openai": {"extra_body": {"reasoning": {"effort": "medium"}}}},
        )

        await provider.complete(request)

        self.assertEqual(
            client.responses.calls[0]["extra_body"],
            {"reasoning": {"effort": "medium"}},
        )

    async def test_capabilities_are_conservative(self) -> None:
        from runlet.providers.openai import OpenAIResponsesProvider

        provider = OpenAIResponsesProvider(model="gpt-test", client=RecordingClient(FakeOpenAIResponse("ok")))

        capabilities = await provider.capabilities()

        self.assertEqual(capabilities.model_name, "gpt-test")
        self.assertFalse(capabilities.supports_tools)
        self.assertFalse(capabilities.supports_parallel_tool_calls)
        self.assertTrue(capabilities.supports_streaming)
        self.assertGreater(capabilities.context_window, 0)

    async def test_tool_messages_are_rejected(self) -> None:
        from runlet.providers.openai import OpenAIResponsesProvider

        provider = OpenAIResponsesProvider(model="gpt-test", client=RecordingClient(FakeOpenAIResponse("ok")))
        request = ModelRequest(messages=[Message.tool("tool output")])

        with self.assertRaises(ValueError):
            await provider.complete(request)

    async def test_stream_yields_text_deltas(self) -> None:
        from runlet.providers.openai import OpenAIResponsesProvider

        client = RecordingClient(FakeOpenAIResponse("ok"))
        client.responses.stream_events = [
            types.SimpleNamespace(type="response.created"),
            types.SimpleNamespace(type="response.output_text.delta", delta="hel"),
            types.SimpleNamespace(type="response.output_text.delta", delta="lo"),
            types.SimpleNamespace(
                type="response.completed",
                response=types.SimpleNamespace(
                    usage=types.SimpleNamespace(input_tokens=3, output_tokens=2),
                ),
            ),
        ]
        provider = OpenAIResponsesProvider(model="gpt-test", client=client)

        events = [event async for event in provider.stream(ModelRequest(messages=[Message.user("hi")]))]

        self.assertEqual([event.kind for event in events], ["text_delta", "text_delta", "usage", "message_completed", "completed"])
        self.assertEqual([event.delta for event in events if event.kind == "text_delta"], ["hel", "lo"])
        self.assertEqual(events[2].usage, Usage(input_tokens=3, output_tokens=2))

    async def test_stream_supports_context_managed_response_stream(self) -> None:
        from runlet.providers.openai import OpenAIResponsesProvider

        client = ContextManagedClient(FakeOpenAIResponse("ok"))
        client.responses.stream_events = [
            types.SimpleNamespace(type="response.output_text.delta", delta="hel"),
            types.SimpleNamespace(type="response.output_text.delta", delta="lo"),
            types.SimpleNamespace(type="response.completed"),
        ]
        provider = OpenAIResponsesProvider(model="gpt-test", client=client)

        events = [event async for event in provider.stream(ModelRequest(messages=[Message.user("hi")]))]

        self.assertEqual([event.delta for event in events if event.kind == "text_delta"], ["hel", "lo"])
        manager = client.responses.last_manager
        assert manager is not None
        self.assertTrue(manager.entered)
        self.assertTrue(manager.exited)

    async def test_stream_yields_tool_call_events(self) -> None:
        from runlet.providers.openai import OpenAIResponsesProvider

        client = RecordingClient(FakeOpenAIResponse("ok"))
        client.responses.stream_events = [
            types.SimpleNamespace(
                type="response.output_item.added",
                item=types.SimpleNamespace(id="call_1", type="function_call", name="lookup"),
            ),
            types.SimpleNamespace(
                type="response.function_call_arguments.delta",
                item_id="call_1",
                delta='{"order_id":"12',
            ),
            types.SimpleNamespace(
                type="response.function_call_arguments.done",
                item_id="call_1",
                arguments='{"order_id":"123"}',
            ),
            types.SimpleNamespace(type="response.completed"),
        ]
        provider = OpenAIResponsesProvider(model="gpt-test", client=client)

        events = [event async for event in provider.stream(ModelRequest(messages=[Message.user("hi")]))]

        self.assertEqual(events[0].kind, "tool_call_delta")
        self.assertEqual(events[0].call_id, "call_1")
        self.assertEqual(events[0].name, "lookup")
        self.assertEqual(events[0].arguments_delta, '{"order_id":"12')
        self.assertEqual(events[1].kind, "tool_call_completed")
        self.assertEqual(events[1].arguments, {"order_id": "123"})

    async def test_stream_forwards_openai_extra_body_options(self) -> None:
        from runlet.providers.openai import OpenAIResponsesProvider

        client = RecordingClient(FakeOpenAIResponse("ok"))
        client.responses.stream_events = []
        provider = OpenAIResponsesProvider(model="gpt-test", client=client)
        request = ModelRequest(
            messages=[Message.user("Hi")],
            options={"openai": {"extra_body": {"reasoning": {"effort": "low"}}}},
        )

        _ = [event async for event in provider.stream(request)]

        self.assertEqual(
            client.responses.calls[0]["extra_body"],
            {"reasoning": {"effort": "low"}},
        )

    def test_missing_openai_dependency_raises_helpful_error(self) -> None:
        sys.modules.pop("openai", None)

        with patch.dict(sys.modules, {"openai": None}):
            from runlet.providers.openai import OpenAIResponsesProvider

            with self.assertRaisesRegex(RuntimeError, "pip install .*openai"):
                OpenAIResponsesProvider(model="gpt-test")

    def test_provider_passes_base_url_to_sdk_client(self) -> None:
        recording_calls: list[dict[str, object]] = []

        class FakeSDKClient:
            def __init__(self, **kwargs: object) -> None:
                recording_calls.append(kwargs)
                self.responses = RecordingResponsesAPI(FakeOpenAIResponse("ok"))

        fake_module = types.SimpleNamespace(OpenAI=FakeSDKClient)
        sys.modules.pop("openai", None)

        with patch("runlet.providers.openai.import_module", return_value=fake_module):
            from runlet.providers.openai import OpenAIResponsesProvider

            OpenAIResponsesProvider(
                model="gpt-test",
                api_key="secret",
                base_url="https://example.test/v1",
            )

        self.assertEqual(
            recording_calls[0],
            {"api_key": "secret", "base_url": "https://example.test/v1"},
        )
