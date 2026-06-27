from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from runlet.core.errors import ContextOverflowError
from runlet.core.models import ModelCapabilities, ModelRequest
from runlet.runtime.policies import ContextPolicy


@dataclass(frozen=True)
class TokenEstimate:
    input_tokens: int
    source: str = "estimated"


@dataclass(frozen=True)
class PreparedContext:
    request: ModelRequest
    estimate: TokenEstimate


class TokenEstimator(Protocol):
    def estimate_request(self, request: ModelRequest) -> TokenEstimate:
        ...


class SimpleTokenEstimator:
    def estimate_request(self, request: ModelRequest) -> TokenEstimate:
        text = " ".join(message.text for message in request.messages)
        tool_tokens = sum(len(getattr(tool, "name", "").split()) + 8 for tool in request.tools)
        return TokenEstimate(input_tokens=len(text.split()) + tool_tokens)


class ContextManager:
    def __init__(self, estimator: TokenEstimator | None = None, policy: ContextPolicy | None = None) -> None:
        self.estimator = estimator or SimpleTokenEstimator()
        self.policy = policy or ContextPolicy()

    async def prepare(self, request: ModelRequest, capabilities: ModelCapabilities) -> PreparedContext:
        estimate = self.estimator.estimate_request(request)
        limit = self.policy.max_context_tokens or capabilities.context_window
        available_input = limit - self.policy.reserved_output_tokens
        if estimate.input_tokens > available_input:
            raise ContextOverflowError(
                f"Context estimate {estimate.input_tokens} exceeds available input budget {available_input}"
            )
        return PreparedContext(request=request, estimate=estimate)
