import unittest

import runlet


class PublicApiTests(unittest.TestCase):
    def test_expected_public_api_exports(self) -> None:
        expected = {
            "Agent",
            "Runtime",
            "Message",
            "ModelResponse",
            "ToolCall",
            "ToolSpec",
            "tool",
            "RuntimeEvent",
            "InMemoryObserver",
            "ContextManager",
            "BaseHook",
            "InMemoryStateStore",
        }

        self.assertTrue(expected.issubset(set(runlet.__all__)))
