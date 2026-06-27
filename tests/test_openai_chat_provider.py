import sys
import types
import unittest
from unittest.mock import patch

from runlet.core import Message
from runlet.core.models import ModelRequest
from runlet.core.runs import Usage


class FakeChatResponse:
    def __init__(
        self,
        content: str | None,
        *,
        tool_calls: list[object] | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self.choices = [
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content=content,
                    tool_calls=tool_calls,
                )
            )
        ]
        self.usage = types.SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens)


class RecordingChatCompletionsAPI:
    def __init__(self, response: FakeChatResponse) -> None:
        self._response = response
        self.calls: list[dict[str, object]] = []
        self.stream_events: list[object] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return self.stream_events
        return self._response


class RecordingClient:
    def __init__(self, response: FakeChatResponse) -> None:
        self.chat = types.SimpleNamespace(completions=RecordingChatCompletionsAPI(response))


class OpenAIChatProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_maps_messages_and_returns_model_response(self) -> None:
        from runlet.providers.openai_chat import OpenAIChatCompletionsProvider

        fake_response = FakeChatResponse("hello back", input_tokens=5, output_tokens=7)
        client = RecordingClient(fake_response)
        provider = OpenAIChatCompletionsProvider(model="gpt-test", client=client)
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
        self.assertEqual(client.chat.completions.calls[0]["model"], "gpt-test")
        self.assertEqual(
            client.chat.completions.calls[0]["messages"],
            [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Previous reply"},
            ],
        )

    async def test_provider_maps_tools_tool_messages_and_options(self) -> None:
        from runlet.providers.openai_chat import OpenAIChatCompletionsProvider

        fake_response = FakeChatResponse("ok")
        client = RecordingClient(fake_response)
        provider = OpenAIChatCompletionsProvider(model="gpt-test", client=client)
        request = ModelRequest(
            messages=[
                Message.system("Be helpful."),
                Message.user("Check order 123."),
                Message.assistant(
                    "",
                    metadata={
                        "tool_calls": [
                            {"id": "call_1", "name": "lookup_order", "arguments": {"order_id": "123"}}
                        ]
                    },
                ),
                Message.tool("order 123 shipped", metadata={"tool_call_id": "call_1", "name": "lookup_order"}),
            ],
            tools=[
                types.SimpleNamespace(
                    name="lookup_order",
                    description="Look up an order.",
                    input_schema={
                        "type": "object",
                        "required": ["order_id"],
                        "properties": {"order_id": {"type": "string"}},
                    },
                )
            ],
            options={
                "openai_chat": {
                    "temperature": 0.1,
                    "max_tokens": 100,
                    "extra_body": {"store": False},
                    "extra_headers": {"x-test": "1"},
                }
            },
        )

        await provider.complete(request)

        call = client.chat.completions.calls[0]
        self.assertEqual(call["temperature"], 0.1)
        self.assertEqual(call["max_tokens"], 100)
        self.assertEqual(call["extra_body"], {"store": False})
        self.assertEqual(call["extra_headers"], {"x-test": "1"})
        self.assertEqual(
            call["tools"],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_order",
                        "description": "Look up an order.",
                        "parameters": {
                            "type": "object",
                            "required": ["order_id"],
                            "properties": {"order_id": {"type": "string"}},
                        },
                    },
                }
            ],
        )
        self.assertEqual(
            call["messages"][-2:],
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup_order",
                                "arguments": '{"order_id":"123"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "order 123 shipped",
                },
            ],
        )

    async def test_complete_parses_tool_calls(self) -> None:
        from runlet.providers.openai_chat import OpenAIChatCompletionsProvider

        fake_response = FakeChatResponse(
            None,
            tool_calls=[
                types.SimpleNamespace(
                    id="call_1",
                    function=types.SimpleNamespace(name="lookup_order", arguments='{"order_id":"123"}'),
                )
            ],
        )
        provider = OpenAIChatCompletionsProvider(model="gpt-test", client=RecordingClient(fake_response))

        response = await provider.complete(ModelRequest(messages=[Message.user("Check order 123.")]))

        self.assertEqual(response.tool_calls[0].id, "call_1")
        self.assertEqual(response.tool_calls[0].name, "lookup_order")
        self.assertEqual(response.tool_calls[0].arguments, {"order_id": "123"})

    async def test_stream_yields_text_deltas(self) -> None:
        from runlet.providers.openai_chat import OpenAIChatCompletionsProvider

        client = RecordingClient(FakeChatResponse("ok"))
        client.chat.completions.stream_events = [
            types.SimpleNamespace(
                choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content="hel"), finish_reason=None)],
                usage=None,
            ),
            types.SimpleNamespace(
                choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content="lo"), finish_reason=None)],
                usage=None,
            ),
            types.SimpleNamespace(
                choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=None), finish_reason="stop")],
                usage=types.SimpleNamespace(prompt_tokens=3, completion_tokens=2),
            ),
        ]
        provider = OpenAIChatCompletionsProvider(model="gpt-test", client=client)

        events = [event async for event in provider.stream(ModelRequest(messages=[Message.user("hi")]))]

        self.assertEqual(
            [event.kind for event in events],
            ["text_delta", "text_delta", "usage", "message_completed", "completed"],
        )
        self.assertEqual([event.delta for event in events if event.kind == "text_delta"], ["hel", "lo"])
        self.assertEqual(events[2].usage, Usage(input_tokens=3, output_tokens=2))

    async def test_stream_yields_tool_call_events(self) -> None:
        from runlet.providers.openai_chat import OpenAIChatCompletionsProvider

        client = RecordingClient(FakeChatResponse("ok"))
        client.chat.completions.stream_events = [
            types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        delta=types.SimpleNamespace(
                            content=None,
                            tool_calls=[
                                types.SimpleNamespace(
                                    index=0,
                                    id="call_1",
                                    function=types.SimpleNamespace(name="lookup", arguments='{"order_id":"12'),
                                )
                            ],
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        delta=types.SimpleNamespace(
                            content=None,
                            tool_calls=[
                                types.SimpleNamespace(
                                    index=0,
                                    id=None,
                                    function=types.SimpleNamespace(name=None, arguments='3"}'),
                                )
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=None,
            ),
        ]
        provider = OpenAIChatCompletionsProvider(model="gpt-test", client=client)

        events = [event async for event in provider.stream(ModelRequest(messages=[Message.user("hi")]))]

        self.assertEqual(events[0].kind, "tool_call_delta")
        self.assertEqual(events[0].call_id, "call_1")
        self.assertEqual(events[0].name, "lookup")
        self.assertEqual(events[0].arguments_delta, '{"order_id":"12')
        self.assertEqual(events[1].kind, "tool_call_delta")
        self.assertEqual(events[1].arguments_delta, '3"}')
        self.assertEqual(events[2].kind, "tool_call_completed")
        self.assertEqual(events[2].arguments, {"order_id": "123"})

    async def test_stream_forwards_openai_chat_extra_body_options(self) -> None:
        from runlet.providers.openai_chat import OpenAIChatCompletionsProvider

        client = RecordingClient(FakeChatResponse("ok"))
        client.chat.completions.stream_events = []
        provider = OpenAIChatCompletionsProvider(model="gpt-test", client=client)
        request = ModelRequest(
            messages=[Message.user("Hi")],
            options={"openai_chat": {"extra_body": {"store": False}}},
        )

        _ = [event async for event in provider.stream(request)]

        self.assertEqual(
            client.chat.completions.calls[0]["extra_body"],
            {"store": False},
        )

    async def test_capabilities_are_conservative(self) -> None:
        from runlet.providers.openai_chat import OpenAIChatCompletionsProvider

        provider = OpenAIChatCompletionsProvider(model="gpt-test", client=RecordingClient(FakeChatResponse("ok")))

        capabilities = await provider.capabilities()

        self.assertEqual(capabilities.model_name, "gpt-test")
        self.assertTrue(capabilities.supports_tools)
        self.assertFalse(capabilities.supports_parallel_tool_calls)
        self.assertTrue(capabilities.supports_streaming)
        self.assertGreater(capabilities.context_window, 0)

    def test_missing_openai_dependency_raises_helpful_error(self) -> None:
        sys.modules.pop("openai", None)

        with patch.dict(sys.modules, {"openai": None}):
            from runlet.providers.openai_chat import OpenAIChatCompletionsProvider

            with self.assertRaisesRegex(RuntimeError, "pip install .*openai"):
                OpenAIChatCompletionsProvider(model="gpt-test")

    def test_provider_passes_base_url_to_sdk_client(self) -> None:
        recording_calls: list[dict[str, object]] = []

        class FakeSDKClient:
            def __init__(self, **kwargs: object) -> None:
                recording_calls.append(kwargs)
                self.chat = types.SimpleNamespace(completions=RecordingChatCompletionsAPI(FakeChatResponse("ok")))

        fake_module = types.SimpleNamespace(OpenAI=FakeSDKClient)
        sys.modules.pop("openai", None)

        with patch("runlet.providers.openai_chat.import_module", return_value=fake_module):
            from runlet.providers.openai_chat import OpenAIChatCompletionsProvider

            OpenAIChatCompletionsProvider(
                model="gpt-test",
                api_key="secret",
                base_url="https://example.test/v1",
            )

        self.assertEqual(
            recording_calls[0],
            {"api_key": "secret", "base_url": "https://example.test/v1"},
        )
