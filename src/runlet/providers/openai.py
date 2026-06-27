from __future__ import annotations

from importlib import import_module
from typing import Any

from runlet.core.messages import Message
from runlet.core.models import ModelCapabilities, ModelRequest, ModelResponse, ModelStreamEvent
from runlet.core.runs import Usage


class OpenAIResponsesProvider:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._client = client or self._build_client(api_key=api_key)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        payload = self._build_input(request)
        response = self._client.responses.create(model=self.model, input=payload)
        return ModelResponse(
            message=Message.assistant(response.output_text),
            usage=self._usage_from_response(response),
            raw=response,
        )

    async def stream(self, request: ModelRequest):
        del request
        raise NotImplementedError("OpenAIResponsesProvider.stream is not implemented yet")
        yield ModelStreamEvent()

    async def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_name=self.model,
            context_window=128_000,
            supports_tools=False,
            supports_parallel_tool_calls=False,
            supports_streaming=False,
        )

    def _build_client(self, api_key: str | None) -> Any:
        try:
            openai_module = import_module("openai")
        except ImportError as exc:
            raise RuntimeError(
                "OpenAIResponsesProvider requires the 'openai' package. Install it with 'pip install openai'."
            ) from exc
        return openai_module.OpenAI(api_key=api_key)

    def _build_input(self, request: ModelRequest) -> list[dict[str, str]]:
        payload: list[dict[str, str]] = []
        for message in request.messages:
            if message.role == "tool":
                raise ValueError("OpenAIResponsesProvider does not support tool messages in v1")
            payload.append({"role": message.role, "content": message.text})
        return payload

    def _usage_from_response(self, response: Any) -> Usage:
        usage = getattr(response, "usage", None)
        if usage is None:
            return Usage()
        return Usage(
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        )
