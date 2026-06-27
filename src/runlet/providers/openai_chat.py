from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from importlib import import_module
import json
from typing import Any, ContextManager, cast

from runlet.core.messages import Message, ToolCall
from runlet.core.models import ModelCapabilities, ModelRequest, ModelResponse, ProviderStreamEvent
from runlet.core.runs import Usage


class OpenAIChatCompletionsProvider:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._client = client or self._build_client(api_key=api_key, base_url=base_url)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        response = self._client.chat.completions.create(**self._build_create_kwargs(request))
        message = self._message_from_response(response)
        return ModelResponse(
            message=message,
            tool_calls=self._tool_calls_from_message(getattr(self._first_choice(response), "message", None)),
            reasoning=self._reasoning_from_message(getattr(self._first_choice(response), "message", None)),
            usage=self._usage_from_response(response),
            raw=response,
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ProviderStreamEvent]:
        stream_result = self._client.chat.completions.create(**self._build_create_kwargs(request, stream=True))
        pending_calls: dict[int, dict[str, str]] = {}
        saw_message_completed = False

        for chunk in self._iter_stream_events(stream_result):
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                usage_event = self._usage_from_chat_usage(usage)
                if usage_event.total_tokens > 0:
                    yield ProviderStreamEvent.usage_event(usage_event, raw=chunk)

            choice = self._first_choice(chunk)
            if choice is None:
                continue

            delta = getattr(choice, "delta", None)
            reasoning_delta = self._reasoning_from_message(delta)
            if reasoning_delta:
                yield ProviderStreamEvent.reasoning_delta(reasoning_delta, raw=chunk)
            text_delta = self._text_from_content(getattr(delta, "content", None))
            if text_delta:
                yield ProviderStreamEvent.text_delta(text_delta, raw=chunk)

            for tool_delta in cast(list[Any], getattr(delta, "tool_calls", None) or []):
                index = int(getattr(tool_delta, "index", 0) or 0)
                call = pending_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                tool_id = str(getattr(tool_delta, "id", "") or "")
                if tool_id:
                    call["id"] = tool_id
                function = getattr(tool_delta, "function", None)
                tool_name = str(getattr(function, "name", "") or "")
                if tool_name:
                    call["name"] = tool_name
                arguments_delta = str(getattr(function, "arguments", "") or "")
                if arguments_delta:
                    call["arguments"] += arguments_delta
                    yield ProviderStreamEvent.tool_call_delta(
                        call_id=call["id"],
                        name=call["name"] or None,
                        arguments_delta=arguments_delta,
                        raw=chunk,
                    )

            finish_reason = str(getattr(choice, "finish_reason", "") or "")
            if finish_reason == "tool_calls":
                for call in pending_calls.values():
                    if call["id"] and call["name"]:
                        yield ProviderStreamEvent.tool_call_completed(
                            call_id=call["id"],
                            name=call["name"],
                            arguments=self._parse_arguments(call["arguments"]),
                            raw=chunk,
                        )
                pending_calls.clear()
                continue

            if finish_reason == "stop":
                saw_message_completed = True

        if saw_message_completed:
            yield ProviderStreamEvent.message_completed()
        yield ProviderStreamEvent.completed()

    async def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_name=self.model,
            context_window=128_000,
            supports_tools=True,
            supports_parallel_tool_calls=False,
            supports_streaming=True,
        )

    def _build_client(self, api_key: str | None, base_url: str | None) -> Any:
        try:
            openai_module = import_module("openai")
        except ImportError as exc:
            raise RuntimeError(
                "OpenAIChatCompletionsProvider requires the 'openai' package. Install it with 'pip install openai'."
            ) from exc
        return openai_module.OpenAI(api_key=api_key, base_url=base_url)

    def _build_create_kwargs(self, request: ModelRequest, *, stream: bool = False) -> dict[str, Any]:
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._build_messages(request),
        }
        if request.tools:
            create_kwargs["tools"] = self._build_tools(request)
        if stream:
            create_kwargs["stream"] = True

        chat_options = cast(dict[str, Any], request.options.get("openai_chat", {}))
        if "temperature" in chat_options:
            create_kwargs["temperature"] = chat_options["temperature"]
        if "max_tokens" in chat_options:
            create_kwargs["max_tokens"] = chat_options["max_tokens"]
        extra_body = chat_options.get("extra_body")
        if extra_body is not None:
            create_kwargs["extra_body"] = extra_body
        extra_headers = chat_options.get("extra_headers")
        if extra_headers is not None:
            create_kwargs["extra_headers"] = extra_headers
        return create_kwargs

    def _build_messages(self, request: ModelRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == "tool":
                messages.append(self._tool_message_item(message))
                continue

            tool_calls = message.metadata.get("tool_calls")
            if message.role == "assistant" and isinstance(tool_calls, list):
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.text,
                        "tool_calls": self._assistant_tool_calls(cast(list[dict[str, Any]], tool_calls)),
                    }
                )
                continue

            messages.append({"role": message.role, "content": message.text})
        return messages

    def _build_tools(self, request: ModelRequest) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for tool in request.tools:
            name = getattr(tool, "name", "")
            if not name:
                continue
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": str(name),
                        "description": str(getattr(tool, "description", "") or ""),
                        "parameters": cast(dict[str, Any], getattr(tool, "input_schema", {})),
                    },
                }
            )
        return tools

    def _assistant_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            items.append(
                {
                    "id": str(tool_call.get("id", "") or ""),
                    "type": "function",
                    "function": {
                        "name": str(tool_call.get("name", "") or ""),
                        "arguments": json.dumps(tool_call.get("arguments", {}), separators=(",", ":")),
                    },
                }
            )
        return items

    def _tool_message_item(self, message: Message) -> dict[str, Any]:
        tool_call_id = str(message.metadata.get("tool_call_id", "") or "")
        if not tool_call_id:
            raise ValueError("OpenAIChatCompletionsProvider tool messages require metadata.tool_call_id")
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": message.text,
        }

    def _message_from_response(self, response: Any) -> Message:
        choice = self._first_choice(response)
        message = getattr(choice, "message", None)
        return Message.assistant(self._text_from_content(getattr(message, "content", None)))

    def _tool_calls_from_message(self, message: Any) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        for item in cast(list[Any], getattr(message, "tool_calls", None) or []):
            function = getattr(item, "function", None)
            tool_calls.append(
                ToolCall(
                    id=str(getattr(item, "id", "") or ""),
                    name=str(getattr(function, "name", "") or ""),
                    arguments=self._parse_arguments(getattr(function, "arguments", "")),
                )
            )
        return tool_calls

    def _reasoning_from_message(self, message: Any) -> str:
        if message is None:
            return ""
        for attr in ("reasoning_content", "reasoning", "thinking"):
            value = getattr(message, attr, None)
            if isinstance(value, str) and value:
                return value
        return ""

    def _usage_from_response(self, response: Any) -> Usage:
        return self._usage_from_chat_usage(getattr(response, "usage", None))

    def _usage_from_chat_usage(self, usage: Any) -> Usage:
        if usage is None:
            return Usage()
        return Usage(
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )

    def _parse_arguments(self, arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return cast(dict[str, Any], arguments.copy())
        if not isinstance(arguments, str) or not arguments:
            return {}
        return cast(dict[str, Any], json.loads(arguments))

    def _text_from_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in cast(list[object], content):
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(item, dict):
                    item_dict = cast(dict[str, Any], item)
                    item_text = item_dict.get("text")
                    if isinstance(item_text, str):
                        parts.append(item_text)
            return "".join(parts)
        return ""

    def _first_choice(self, response: Any) -> Any | None:
        choices = getattr(response, "choices", None)
        if not choices:
            return None
        return choices[0]

    def _iter_stream_events(self, stream_result: object) -> Iterable[object]:
        if hasattr(stream_result, "__enter__") and hasattr(stream_result, "__exit__"):
            with cast(ContextManager[Iterable[object]], stream_result) as managed_stream:
                yield from managed_stream
            return
        yield from cast(Iterable[object], stream_result)
