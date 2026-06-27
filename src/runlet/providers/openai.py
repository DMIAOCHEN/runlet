from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from importlib import import_module
from typing import Any, Callable, cast

from runlet.core.messages import Message
from runlet.core.models import ModelCapabilities, ModelRequest, ModelResponse, ModelStreamEvent
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
            usage=self._usage_from_response(response),
            raw=response,
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        stream_method = cast(
            Callable[..., Iterable[object]] | None,
            getattr(self._client.responses, "stream", None),
        )
        if callable(stream_method):
            stream: Iterable[object] = stream_method(**self._build_create_kwargs(request))
        else:
            stream = cast(
                Iterable[object],
                self._client.responses.create(**self._build_create_kwargs(request), stream=True),
            )

        for event in stream:
            event_type = getattr(event, "type", "")
            if event_type == "response.output_text.delta":
                yield ModelStreamEvent(delta=str(getattr(event, "delta", "")))
                continue

            if event_type == "response.completed":
                response = getattr(event, "response", None)
                yield ModelStreamEvent(final=True, usage=self._usage_from_response(response))

    async def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_name=self.model,
            context_window=128_000,
            supports_tools=False,
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

    def _build_input(self, request: ModelRequest) -> list[dict[str, str]]:
        payload: list[dict[str, str]] = []
        for message in request.messages:
            if message.role == "tool":
                raise ValueError("OpenAIResponsesProvider does not support tool messages in v1")
            payload.append({"role": message.role, "content": message.text})
        return payload

    def _build_create_kwargs(self, request: ModelRequest) -> dict[str, Any]:
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "input": self._build_input(request),
        }
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
