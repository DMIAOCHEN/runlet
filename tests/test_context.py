import unittest

from runlet.context import ContextManager, SimpleTokenEstimator
from runlet.core import Message
from runlet.errors import ContextOverflowError
from runlet.models import ModelCapabilities, ModelRequest
from runlet.policies import ContextPolicy


class ContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_manager_allows_request_under_budget(self) -> None:
        manager = ContextManager(SimpleTokenEstimator(), ContextPolicy(reserved_output_tokens=5))
        request = ModelRequest(messages=[Message.user("hello world")])

        prepared = await manager.prepare(request, ModelCapabilities(model_name="fake", context_window=100))

        self.assertIs(prepared.request, request)
        self.assertLess(prepared.estimate.input_tokens, 100)

    async def test_context_manager_fails_closed_when_over_budget(self) -> None:
        manager = ContextManager(SimpleTokenEstimator(), ContextPolicy(reserved_output_tokens=5))
        request = ModelRequest(messages=[Message.user("one two three four five six seven eight nine ten")])

        with self.assertRaises(ContextOverflowError):
            await manager.prepare(request, ModelCapabilities(model_name="tiny", context_window=8))
