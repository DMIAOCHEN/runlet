from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from importlib import import_module
import json
from typing import Any, Callable, ContextManager, cast

from runlet.core.messages import Message, ToolCall
from runlet.core.models import ModelCapabilities, ModelRequest, ModelResponse, ProviderStreamEvent
from runlet.core.runs import Usage


class AnthropicMessagesProvider:
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
        response = self._client.messages.create(**self._build_create_kwargs(request))
        return ModelResponse(
            message=Message.assistant(self._text_from_content(response.content)),
            tool_calls=self._tool_calls_from_content(response.content),
            reasoning=self._reasoning_from_content(response.content),
            usage=self._usage_from_response(response),
            raw=response,
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ProviderStreamEvent]:
        stream_method = cast(Callable[..., object] | None, getattr(self._client.messages, "stream", None))
        if callable(stream_method):
            stream_result = stream_method(**self._build_create_kwargs(request))
        else:
            stream_result = cast(
                Iterable[object],
                self._client.messages.create(**self._build_create_kwargs(request, stream=True)),
            )

        tool_blocks: dict[int, dict[str, str]] = {}
        completed_tool_calls: set[str] = set()
        input_tokens = 0
        output_tokens = 0

        for event in self._iter_stream_events(stream_result):
            event_type = str(getattr(event, "type", "") or "")

            if event_type == "message_start":
                message = getattr(event, "message", None)
                usage = getattr(message, "usage", None)
                input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
                output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
                continue

            if event_type == "content_block_start":
                index = int(getattr(event, "index", 0) or 0)
                block = getattr(event, "content_block", None)
                if getattr(block, "type", "") == "tool_use":
                    block_input = getattr(block, "input", None)
                    tool_blocks[index] = {
                        "id": str(getattr(block, "id", "") or ""),
                        "name": str(getattr(block, "name", "") or ""),
                        "arguments": (
                            json.dumps(block_input, separators=(",", ":"))
                            if isinstance(block_input, dict) and block_input
                            else ""
                        ),
                    }
                continue

            if event_type == "content_block_delta":
                index = int(getattr(event, "index", 0) or 0)
                delta = getattr(event, "delta", None)
                delta_type = str(getattr(delta, "type", "") or "")

                if delta_type == "text_delta":
                    text_delta = str(getattr(delta, "text", "") or "")
                    if text_delta:
                        yield ProviderStreamEvent.text_delta(text_delta, raw=event)
                    continue

                if delta_type == "thinking_delta":
                    reasoning_delta = str(getattr(delta, "thinking", "") or "")
                    if reasoning_delta:
                        yield ProviderStreamEvent.reasoning_delta(reasoning_delta, raw=event)
                    continue

                if delta_type == "input_json_delta":
                    arguments_delta = str(getattr(delta, "partial_json", "") or "")
                    if not arguments_delta:
                        continue
                    tool_block = tool_blocks.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    tool_block["arguments"] += arguments_delta
                    if tool_block["id"]:
                        yield ProviderStreamEvent.tool_call_delta(
                            call_id=tool_block["id"],
                            name=tool_block["name"] or None,
                            arguments_delta=arguments_delta,
                            raw=event,
                        )
                continue

            if event_type == "content_block_stop":
                index = int(getattr(event, "index", 0) or 0)
                tool_block = tool_blocks.get(index)
                if tool_block is None:
                    continue
                call_id = tool_block["id"]
                name = tool_block["name"]
                if call_id and name and call_id not in completed_tool_calls:
                    completed_tool_calls.add(call_id)
                    yield ProviderStreamEvent.tool_call_completed(
                        call_id=call_id,
                        name=name,
                        arguments=self._parse_arguments(tool_block["arguments"]),
                        raw=event,
                    )
                continue

            if event_type == "message_delta":
                usage = getattr(event, "usage", None)
                output_tokens = int(getattr(usage, "output_tokens", output_tokens) or output_tokens)
                continue

            if event_type == "message_stop":
                usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)
                if usage.total_tokens > 0:
                    yield ProviderStreamEvent.usage_event(usage, raw=event)
                yield ProviderStreamEvent.message_completed(raw=event)
                yield ProviderStreamEvent.completed(raw=event)

    async def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_name=self.model,
            context_window=200_000,
            supports_tools=True,
            supports_parallel_tool_calls=False,
            supports_streaming=True,
        )

    def _build_client(self, api_key: str | None, base_url: str | None) -> Any:
        try:
            anthropic_module = import_module("anthropic")
        except ImportError as exc:
            raise RuntimeError(
                "AnthropicMessagesProvider requires the 'anthropic' package. Install it with 'pip install anthropic'."
            ) from exc
        return anthropic_module.Anthropic(api_key=api_key, base_url=base_url)

    def _build_create_kwargs(self, request: ModelRequest, *, stream: bool = False) -> dict[str, Any]:
        anthropic_options = cast(dict[str, Any], request.options.get("anthropic", {}))
        system, messages = self._split_system_messages(request)
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(anthropic_options.get("max_tokens", 1024)),
        }
        if system:
            create_kwargs["system"] = system
        if request.tools:
            create_kwargs["tools"] = self._build_tools(request)
        if stream:
            create_kwargs["stream"] = True
        if "temperature" in anthropic_options:
            create_kwargs["temperature"] = anthropic_options["temperature"]
        if "thinking" in anthropic_options:
            create_kwargs["thinking"] = anthropic_options["thinking"]
        if "metadata" in anthropic_options:
            create_kwargs["metadata"] = anthropic_options["metadata"]
        if "stop_sequences" in anthropic_options:
            create_kwargs["stop_sequences"] = anthropic_options["stop_sequences"]
        extra_headers = anthropic_options.get("extra_headers")
        if extra_headers is not None:
            create_kwargs["extra_headers"] = extra_headers
        extra_body = anthropic_options.get("extra_body")
        if extra_body is not None:
            create_kwargs["extra_body"] = extra_body
        return create_kwargs

    def _split_system_messages(self, request: ModelRequest) -> tuple[str, list[dict[str, Any]]]:
        system_parts: list[str] = []
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == "system":
                if message.text:
                    system_parts.append(message.text)
                continue
            messages.append(self._message_item(message))
        return "\n\n".join(system_parts), messages

    def _message_item(self, message: Message) -> dict[str, Any]:
        if message.role == "tool":
            tool_call_id = str(message.metadata.get("tool_call_id", "") or "")
            if not tool_call_id:
                raise ValueError("AnthropicMessagesProvider tool messages require metadata.tool_call_id")
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": message.text,
                    }
                ],
            }

        tool_calls = message.metadata.get("tool_calls")
        if message.role == "assistant" and isinstance(tool_calls, list):
            content: list[dict[str, Any]] = []
            if message.text:
                content.append({"type": "text", "text": message.text})
            for tool_call in cast(list[dict[str, Any]], tool_calls):
                content.append(
                    {
                        "type": "tool_use",
                        "id": str(tool_call.get("id", "") or ""),
                        "name": str(tool_call.get("name", "") or ""),
                        "input": cast(dict[str, Any], tool_call.get("arguments", {})),
                    }
                )
            return {"role": "assistant", "content": content}

        return {"role": message.role, "content": message.text}

    def _build_tools(self, request: ModelRequest) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for tool in request.tools:
            name = getattr(tool, "name", "")
            if not name:
                continue
            tools.append(
                {
                    "name": str(name),
                    "description": str(getattr(tool, "description", "") or ""),
                    "input_schema": cast(dict[str, Any], getattr(tool, "input_schema", {})),
                }
            )
        return tools

    def _tool_calls_from_content(self, content: object) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        for block in cast(list[object], content or []):
            if getattr(block, "type", "") != "tool_use":
                continue
            tool_calls.append(
                ToolCall(
                    id=str(getattr(block, "id", "") or ""),
                    name=str(getattr(block, "name", "") or ""),
                    arguments=cast(dict[str, Any], getattr(block, "input", {}) or {}),
                )
            )
        return tool_calls

    def _reasoning_from_content(self, content: object) -> str:
        parts: list[str] = []
        for block in cast(list[object], content or []):
            block_type = str(getattr(block, "type", "") or "")
            if block_type == "thinking":
                thinking = getattr(block, "thinking", None)
                if isinstance(thinking, str) and thinking:
                    parts.append(thinking)
            elif block_type == "redacted_thinking":
                text = getattr(block, "text", None)
                if isinstance(text, str) and text:
                    parts.append(text)
        return "".join(parts)

    def _text_from_content(self, content: object) -> str:
        parts: list[str] = []
        for block in cast(list[object], content or []):
            if getattr(block, "type", "") != "text":
                continue
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
        return "".join(parts)

    def _usage_from_response(self, response: Any) -> Usage:
        usage = getattr(response, "usage", None)
        if usage is None:
            return Usage()
        return Usage(
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        )

    def _parse_arguments(self, arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return cast(dict[str, Any], arguments.copy())
        if not isinstance(arguments, str) or not arguments:
            return {}
        return cast(dict[str, Any], json.loads(arguments))

    def _iter_stream_events(self, stream_result: object) -> Iterable[object]:
        if hasattr(stream_result, "__enter__") and hasattr(stream_result, "__exit__"):
            with cast(ContextManager[Iterable[object]], stream_result) as managed_stream:
                yield from managed_stream
            return
        yield from cast(Iterable[object], stream_result)
