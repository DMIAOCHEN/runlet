import sys
import types
import unittest
from typing import cast
from unittest.mock import patch

from runlet.core import Message
from runlet.core.models import ModelRequest
from runlet.core.runs import Usage


class FakeAnthropicResponse:
    def __init__(
        self,
        content: list[object] | None = None,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self.content = content or []
        self.usage = types.SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)


class RecordingMessagesAPI:
    def __init__(self, response: FakeAnthropicResponse) -> None:
        self._response = response
        self.calls: list[dict[str, object]] = []
        self.stream_events: list[object] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return self.stream_events
        return self._response

    def stream(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.stream_events


class RecordingStreamManager:
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


class ContextManagedMessagesAPI(RecordingMessagesAPI):
    def __init__(self, response: FakeAnthropicResponse) -> None:
        super().__init__(response)
        self.last_manager: RecordingStreamManager | None = None

    def stream(self, **kwargs: object) -> RecordingStreamManager:
        self.calls.append(kwargs)
        manager = RecordingStreamManager(self.stream_events)
        self.last_manager = manager
        return manager


class RecordingClient:
    def __init__(self, response: FakeAnthropicResponse) -> None:
        self.messages = RecordingMessagesAPI(response)


class ContextManagedClient:
    def __init__(self, response: FakeAnthropicResponse) -> None:
        self.messages = ContextManagedMessagesAPI(response)


class AnthropicProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_maps_messages_and_returns_model_response(self) -> None:
        from runlet.providers.anthropic import AnthropicMessagesProvider

        fake_response = FakeAnthropicResponse(
            content=[
                types.SimpleNamespace(type="thinking", thinking="internal reasoning"),
                types.SimpleNamespace(type="text", text="hello back"),
            ],
            input_tokens=5,
            output_tokens=7,
        )
        client = RecordingClient(fake_response)
        provider = AnthropicMessagesProvider(model="claude-test", client=client)
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
        self.assertEqual(response.reasoning, "internal reasoning")
        self.assertEqual(response.usage, Usage(input_tokens=5, output_tokens=7))
        self.assertEqual(client.messages.calls[0]["model"], "claude-test")
        self.assertEqual(client.messages.calls[0]["system"], "Be helpful.")
        self.assertEqual(
            client.messages.calls[0]["messages"],
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Previous reply"},
            ],
        )

    async def test_provider_maps_tools_tool_messages_and_options(self) -> None:
        from runlet.providers.anthropic import AnthropicMessagesProvider

        fake_response = FakeAnthropicResponse(content=[types.SimpleNamespace(type="text", text="ok")])
        client = RecordingClient(fake_response)
        provider = AnthropicMessagesProvider(model="claude-test", client=client)
        request = ModelRequest(
            messages=[
                Message.system("Be helpful."),
                Message.user("Check order 123."),
                Message.assistant(
                    "",
                    metadata={
                        "tool_calls": [
                            {"id": "toolu_1", "name": "lookup_order", "arguments": {"order_id": "123"}}
                        ]
                    },
                ),
                Message.tool("order 123 shipped", metadata={"tool_call_id": "toolu_1", "name": "lookup_order"}),
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
                "anthropic": {
                    "max_tokens": 100,
                    "temperature": 0.1,
                    "thinking": {"type": "enabled", "budget_tokens": 256},
                    "metadata": {"user_id": "u1"},
                    "stop_sequences": ["DONE"],
                    "extra_headers": {"x-test": "1"},
                    "extra_body": {"service_tier": "standard"},
                }
            },
        )

        await provider.complete(request)

        call = client.messages.calls[0]
        self.assertEqual(call["max_tokens"], 100)
        self.assertEqual(call["temperature"], 0.1)
        self.assertEqual(call["thinking"], {"type": "enabled", "budget_tokens": 256})
        self.assertEqual(call["metadata"], {"user_id": "u1"})
        self.assertEqual(call["stop_sequences"], ["DONE"])
        self.assertEqual(call["extra_headers"], {"x-test": "1"})
        self.assertEqual(call["extra_body"], {"service_tier": "standard"})
        self.assertEqual(
            call["tools"],
            [
                {
                    "name": "lookup_order",
                    "description": "Look up an order.",
                    "input_schema": {
                        "type": "object",
                        "required": ["order_id"],
                        "properties": {"order_id": {"type": "string"}},
                    },
                }
            ],
        )
        self.assertEqual(
            cast(list[dict[str, object]], call["messages"])[-2:],
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "lookup_order",
                            "input": {"order_id": "123"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "order 123 shipped",
                        }
                    ],
                },
            ],
        )

    async def test_complete_parses_tool_use_blocks(self) -> None:
        from runlet.providers.anthropic import AnthropicMessagesProvider

        fake_response = FakeAnthropicResponse(
            content=[
                types.SimpleNamespace(
                    type="tool_use",
                    id="toolu_1",
                    name="lookup_order",
                    input={"order_id": "123"},
                )
            ]
        )
        provider = AnthropicMessagesProvider(model="claude-test", client=RecordingClient(fake_response))

        response = await provider.complete(ModelRequest(messages=[Message.user("Check order 123.")]))

        self.assertEqual(response.tool_calls[0].id, "toolu_1")
        self.assertEqual(response.tool_calls[0].name, "lookup_order")
        self.assertEqual(response.tool_calls[0].arguments, {"order_id": "123"})

    async def test_stream_yields_text_reasoning_usage_and_completed(self) -> None:
        from runlet.providers.anthropic import AnthropicMessagesProvider

        client = RecordingClient(FakeAnthropicResponse())
        client.messages.stream_events = [
            types.SimpleNamespace(
                type="message_start",
                message=types.SimpleNamespace(usage=types.SimpleNamespace(input_tokens=3, output_tokens=0)),
            ),
            types.SimpleNamespace(type="content_block_delta", delta=types.SimpleNamespace(type="thinking_delta", thinking="think ")),
            types.SimpleNamespace(type="content_block_delta", delta=types.SimpleNamespace(type="text_delta", text="hel")),
            types.SimpleNamespace(type="content_block_delta", delta=types.SimpleNamespace(type="thinking_delta", thinking="more")),
            types.SimpleNamespace(type="content_block_delta", delta=types.SimpleNamespace(type="text_delta", text="lo")),
            types.SimpleNamespace(
                type="message_delta",
                usage=types.SimpleNamespace(output_tokens=2),
            ),
            types.SimpleNamespace(type="message_stop"),
        ]
        provider = AnthropicMessagesProvider(model="claude-test", client=client)

        events = [event async for event in provider.stream(ModelRequest(messages=[Message.user("hi")]))]

        self.assertEqual(
            [event.kind for event in events],
            ["reasoning_delta", "text_delta", "reasoning_delta", "text_delta", "usage", "message_completed", "completed"],
        )
        self.assertEqual([event.delta for event in events if event.kind == "text_delta"], ["hel", "lo"])
        self.assertEqual([event.reasoning for event in events if event.kind == "reasoning_delta"], ["think ", "more"])
        usage_event = next(event for event in events if event.kind == "usage")
        self.assertEqual(usage_event.usage, Usage(input_tokens=3, output_tokens=2))

    async def test_stream_yields_tool_call_events(self) -> None:
        from runlet.providers.anthropic import AnthropicMessagesProvider

        client = RecordingClient(FakeAnthropicResponse())
        client.messages.stream_events = [
            types.SimpleNamespace(
                type="content_block_start",
                index=0,
                content_block=types.SimpleNamespace(type="tool_use", id="toolu_1", name="lookup", input={}),
            ),
            types.SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=types.SimpleNamespace(type="input_json_delta", partial_json='{"order_id":"12'),
            ),
            types.SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=types.SimpleNamespace(type="input_json_delta", partial_json='3"}'),
            ),
            types.SimpleNamespace(type="content_block_stop", index=0),
            types.SimpleNamespace(type="message_stop"),
        ]
        provider = AnthropicMessagesProvider(model="claude-test", client=client)

        events = [event async for event in provider.stream(ModelRequest(messages=[Message.user("hi")]))]

        self.assertEqual(events[0].kind, "tool_call_delta")
        self.assertEqual(events[0].call_id, "toolu_1")
        self.assertEqual(events[0].name, "lookup")
        self.assertEqual(events[0].arguments_delta, '{"order_id":"12')
        self.assertEqual(events[1].kind, "tool_call_delta")
        self.assertEqual(events[1].arguments_delta, '3"}')
        self.assertEqual(events[2].kind, "tool_call_completed")
        self.assertEqual(events[2].arguments, {"order_id": "123"})

    async def test_stream_supports_context_managed_stream(self) -> None:
        from runlet.providers.anthropic import AnthropicMessagesProvider

        client = ContextManagedClient(FakeAnthropicResponse())
        client.messages.stream_events = [types.SimpleNamespace(type="message_stop")]
        provider = AnthropicMessagesProvider(model="claude-test", client=client)

        _ = [event async for event in provider.stream(ModelRequest(messages=[Message.user("hi")]))]

        assert client.messages.last_manager is not None
        self.assertTrue(client.messages.last_manager.entered)
        self.assertTrue(client.messages.last_manager.exited)

    async def test_capabilities_are_conservative(self) -> None:
        from runlet.providers.anthropic import AnthropicMessagesProvider

        provider = AnthropicMessagesProvider(model="claude-test", client=RecordingClient(FakeAnthropicResponse()))

        capabilities = await provider.capabilities()

        self.assertEqual(capabilities.model_name, "claude-test")
        self.assertTrue(capabilities.supports_tools)
        self.assertFalse(capabilities.supports_parallel_tool_calls)
        self.assertTrue(capabilities.supports_streaming)
        self.assertGreater(capabilities.context_window, 0)

    def test_missing_anthropic_dependency_raises_helpful_error(self) -> None:
        sys.modules.pop("anthropic", None)

        with patch.dict(sys.modules, {"anthropic": None}):
            from runlet.providers.anthropic import AnthropicMessagesProvider

            with self.assertRaisesRegex(RuntimeError, "pip install .*anthropic"):
                AnthropicMessagesProvider(model="claude-test")

    def test_provider_passes_base_url_to_sdk_client(self) -> None:
        recording_calls: list[dict[str, object]] = []

        class FakeSDKClient:
            def __init__(self, **kwargs: object) -> None:
                recording_calls.append(kwargs)
                self.messages = RecordingMessagesAPI(FakeAnthropicResponse())

        fake_module = types.SimpleNamespace(Anthropic=FakeSDKClient)
        sys.modules.pop("anthropic", None)

        with patch("runlet.providers.anthropic.import_module", return_value=fake_module):
            from runlet.providers.anthropic import AnthropicMessagesProvider

            AnthropicMessagesProvider(
                model="claude-test",
                api_key="secret",
                base_url="https://example.test",
            )

        self.assertEqual(
            recording_calls[0],
            {"api_key": "secret", "base_url": "https://example.test"},
        )
