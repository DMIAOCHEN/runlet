from __future__ import annotations

from collections.abc import AsyncIterator

from runlet.models import ModelCapabilities, ModelRequest, ModelResponse, ModelStreamEvent


class FakeModelProvider:
    def __init__(
        self,
        responses: list[ModelResponse],
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []
        self._capabilities = capabilities or ModelCapabilities(model_name="fake", context_window=4096)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("FakeModelProvider has no responses left")
        return self.responses.pop(0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        response = await self.complete(request)
        yield ModelStreamEvent(delta=response.message.text, usage=response.usage, final=True)

    async def capabilities(self) -> ModelCapabilities:
        return self._capabilities


class FakeStreamingModelProvider:
    def __init__(
        self,
        deltas: list[str],
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        self.deltas = list(deltas)
        self.requests: list[ModelRequest] = []
        self._capabilities = capabilities or ModelCapabilities(
            model_name="fake-streaming",
            context_window=4096,
            supports_streaming=True,
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError("FakeStreamingModelProvider.complete should not be called")

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        for delta in self.deltas:
            yield ModelStreamEvent(delta=delta)
        yield ModelStreamEvent(final=True)

    async def capabilities(self) -> ModelCapabilities:
        return self._capabilities
