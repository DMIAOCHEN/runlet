import unittest
from typing import Any

from runlet import Agent, Message, Runtime
from runlet.core.models import ModelRequest, ModelResponse
from runlet.integrations import BaseHook, HookRunner
from runlet.testing import FakeModelProvider


class AppendHook(BaseHook):
    async def before_model_request(self, request: ModelRequest, context: Any) -> ModelRequest:
        request.messages.append(Message.user("hooked"))
        return request


class DisabledHook(BaseHook):
    enabled = False

    async def before_model_request(self, request: ModelRequest, context: Any) -> ModelRequest:
        raise AssertionError("disabled hook should not run")


class HookTests(unittest.IsolatedAsyncioTestCase):
    async def test_hook_runner_skips_disabled_hooks(self) -> None:
        runner = HookRunner([DisabledHook()])
        request = ModelRequest(messages=[])

        result = await runner.before_model_request(request, None)

        self.assertIs(result, request)

    async def test_runtime_applies_before_model_request_hook(self) -> None:
        model = FakeModelProvider([ModelResponse(message=Message.assistant("ok"))])
        runtime = Runtime()
        agent = Agent(name="assistant", instructions="Help.", model=model, hooks=(AppendHook(),))

        await runtime.run(agent, "hi")

        self.assertEqual(model.requests[0].messages[-1].text, "hooked")
