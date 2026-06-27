from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from importlib import import_module
import json
from typing import Any, Callable, ContextManager, cast

from runlet.core.messages import Message, ToolCall
from runlet.core.models import ModelCapabilities, ModelRequest, ModelResponse, ProviderStreamEvent
from runlet.core.runs import Usage


class OpenAIResponsesProvider:
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
        response = self._client.responses.create(**self._build_create_kwargs(request))
        return ModelResponse(
            message=Message.assistant(response.output_text),
            tool_calls=self._tool_calls_from_response(response),
            usage=self._usage_from_response(response),
            raw=response,
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ProviderStreamEvent]:
        stream_method = cast(
            Callable[..., object] | None,
            getattr(self._client.responses, "stream", None),
        )
        if callable(stream_method):
            stream_result = stream_method(**self._build_create_kwargs(request))
        else:
            stream_result = cast(
                Iterable[object],
                self._client.responses.create(**self._build_create_kwargs(request), stream=True),
            )

        tool_items: dict[str, dict[str, str]] = {}
        completed_tool_calls: set[str] = set()
        for event in self._iter_stream_events(stream_result):
            event_type = getattr(event, "type", "")
            if event_type == "response.output_text.delta":
                yield ProviderStreamEvent.text_delta(str(getattr(event, "delta", "")), raw=event)
                continue

            if event_type == "response.output_item.added":
                item = getattr(event, "item", None)
                if getattr(item, "type", "") == "function_call":
                    item_id = str(getattr(item, "id", "") or "")
                    item_name = str(getattr(item, "name", "") or "")
                    item_call_id = str(getattr(item, "call_id", "") or item_id)
                    if item_id and item_name and item_call_id:
                        tool_items[item_id] = {"name": item_name, "call_id": item_call_id}
                continue

            if event_type == "response.function_call_arguments.delta":
                call_id = str(getattr(event, "item_id", "") or "")
                item = tool_items.get(call_id, {})
                resolved_call_id = item.get("call_id", call_id)
                if resolved_call_id:
                    yield ProviderStreamEvent.tool_call_delta(
                        call_id=resolved_call_id,
                        name=item.get("name"),
                        arguments_delta=str(getattr(event, "delta", "")),
                        raw=event,
                    )
                continue

            if event_type == "response.function_call_arguments.done":
                call_id = str(getattr(event, "item_id", "") or "")
                item = tool_items.get(call_id, {})
                resolved_call_id = item.get("call_id", call_id)
                if resolved_call_id and resolved_call_id not in completed_tool_calls:
                    completed_tool_calls.add(resolved_call_id)
                    yield ProviderStreamEvent.tool_call_completed(
                        call_id=resolved_call_id,
                        name=item.get("name", ""),
                        arguments=self._parse_arguments(getattr(event, "arguments", "")),
                        raw=event,
                    )
                continue

            if event_type == "response.output_item.done":
                item = getattr(event, "item", None)
                if getattr(item, "type", "") == "function_call":
                    item_id = str(getattr(item, "id", "") or "")
                    call_id = str(getattr(item, "call_id", "") or tool_items.get(item_id, {}).get("call_id", item_id))
                    name = str(getattr(item, "name", "") or tool_items.get(item_id, {}).get("name", ""))
                    if call_id and name and call_id not in completed_tool_calls:
                        completed_tool_calls.add(call_id)
                        yield ProviderStreamEvent.tool_call_completed(
                            call_id=call_id,
                            name=name,
                            arguments=self._parse_arguments(getattr(item, "arguments", "")),
                            raw=event,
                        )
                continue

            if event_type == "response.completed":
                response = getattr(event, "response", None)
                usage = self._usage_from_response(response)
                if usage.total_tokens > 0:
                    yield ProviderStreamEvent.usage_event(usage, raw=event)
                yield ProviderStreamEvent.message_completed(raw=event)
                yield ProviderStreamEvent.completed(raw=event)

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
                "OpenAIResponsesProvider requires the 'openai' package. Install it with 'pip install openai'."
            ) from exc
        return openai_module.OpenAI(api_key=api_key, base_url=base_url)

    def _build_input(self, request: ModelRequest) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == "tool":
                payload.append(self._tool_output_item_from_message(message))
                continue

            tool_calls = message.metadata.get("tool_calls")
            if message.role == "assistant" and isinstance(tool_calls, list):
                if message.text:
                    payload.append({"role": message.role, "content": message.text})
                payload.extend(self._assistant_tool_call_items(cast(list[dict[str, Any]], tool_calls)))
                continue

            payload.append({"role": message.role, "content": message.text})
        return payload

    def _build_create_kwargs(self, request: ModelRequest) -> dict[str, Any]:
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "input": self._build_input(request),
        }
        tools = self._build_tools(request)
        if tools:
            create_kwargs["tools"] = tools
        openai_options = request.options.get("openai", {})
        extra_body = openai_options.get("extra_body")
        if extra_body is not None:
            create_kwargs["extra_body"] = extra_body
        return create_kwargs

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

    def _tool_calls_from_response(self, response: Any) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", "") != "function_call":
                continue
            tool_calls.append(
                ToolCall(
                    id=str(getattr(item, "call_id", "") or getattr(item, "id", "") or ""),
                    name=str(getattr(item, "name", "") or ""),
                    arguments=self._parse_arguments(getattr(item, "arguments", "")),
                )
            )
        return tool_calls

    def _assistant_tool_call_items(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            items.append(
                {
                    "type": "function_call",
                    "call_id": str(tool_call.get("id", "") or ""),
                    "name": str(tool_call.get("name", "") or ""),
                    "arguments": json.dumps(tool_call.get("arguments", {}), separators=(",", ":")),
                }
            )
        return items

    def _tool_output_item_from_message(self, message: Message) -> dict[str, Any]:
        call_id = str(message.metadata.get("tool_call_id", "") or "")
        if not call_id:
            raise ValueError("OpenAIResponsesProvider tool messages require metadata.tool_call_id")
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": message.text,
        }

    def _build_tools(self, request: ModelRequest) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for tool in request.tools:
            name = getattr(tool, "name", "")
            if not name:
                continue
            tools.append(
                {
                    "type": "function",
                    "name": str(name),
                    "description": str(getattr(tool, "description", "") or ""),
                    "parameters": cast(dict[str, Any], getattr(tool, "input_schema", {})),
                }
            )
        return tools

    def _iter_stream_events(self, stream_result: object) -> Iterable[object]:
        if hasattr(stream_result, "__enter__") and hasattr(stream_result, "__exit__"):
            with cast(ContextManager[Iterable[object]], stream_result) as managed_stream:
                yield from managed_stream
            return
        yield from cast(Iterable[object], stream_result)
