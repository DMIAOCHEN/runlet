from __future__ import annotations

from typing import Any


class BaseHook:
    enabled = True

    def is_enabled(self, context: Any) -> bool:
        return bool(self.enabled)

    async def before_model_request(self, request: Any, context: Any) -> Any:
        return request

    async def after_model_response(self, response: Any, context: Any) -> Any:
        return response

    async def before_tool_call(self, call: Any, context: Any) -> Any:
        return call

    async def after_tool_result(self, result: Any, context: Any) -> Any:
        return result


class HookRunner:
    def __init__(self, hooks: list[BaseHook] | tuple[BaseHook, ...]) -> None:
        self.hooks = tuple(hooks)

    def _enabled_hooks(self, context: Any) -> list[BaseHook]:
        return [hook for hook in self.hooks if hook.is_enabled(context)]

    async def before_model_request(self, request: Any, context: Any) -> Any:
        for hook in self._enabled_hooks(context):
            request = await hook.before_model_request(request, context)
        return request

    async def after_model_response(self, response: Any, context: Any) -> Any:
        for hook in self._enabled_hooks(context):
            response = await hook.after_model_response(response, context)
        return response

    async def before_tool_call(self, call: Any, context: Any) -> Any:
        for hook in self._enabled_hooks(context):
            call = await hook.before_tool_call(call, context)
        return call

    async def after_tool_result(self, result: Any, context: Any) -> Any:
        for hook in self._enabled_hooks(context):
            result = await hook.after_tool_result(result, context)
        return result
