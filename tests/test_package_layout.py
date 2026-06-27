import unittest


class PackageLayoutTests(unittest.TestCase):
    def test_new_internal_package_layout_is_importable(self) -> None:
        from runlet.core.messages import Message
        from runlet.integrations.tools import tool
        from runlet.runtime.engine import Runtime
        from runlet.testing.fakes import FakeModelProvider

        self.assertEqual(Message.user("hi").text, "hi")
        self.assertTrue(callable(tool))
        self.assertEqual(Runtime.__name__, "Runtime")
        self.assertEqual(FakeModelProvider.__name__, "FakeModelProvider")
