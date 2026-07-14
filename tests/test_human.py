import unittest

from runlet.core import ToolCall
from runlet.integrations.human import ask_human


class HumanInputToolTests(unittest.TestCase):
    def test_ask_human_converts_valid_choice_call(self) -> None:
        spec = ask_human()

        request = spec.human_request_from_call(
            ToolCall(
                "call_1",
                "ask_human",
                {
                    "kind": "choice",
                    "prompt": "Pick",
                    "options": [
                        {"id": "one", "label": "One"},
                        {"id": "two", "label": "Two"},
                    ],
                },
            )
        )

        self.assertEqual(request.kind, "choice")
        self.assertEqual([option.id for option in request.options], ["one", "two"])
        self.assertTrue(request.id.startswith("hitl_"))
        self.assertIsNone(request.tool_call)

    def test_ask_human_rejects_invalid_kind(self) -> None:
        with self.assertRaises(ValueError):
            ask_human().human_request_from_call(
                ToolCall("call_1", "ask_human", {"kind": "approval", "prompt": "Continue?"})
            )

    def test_ask_human_rejects_empty_choice_list(self) -> None:
        with self.assertRaises(ValueError):
            ask_human().human_request_from_call(
                ToolCall("call_1", "ask_human", {"kind": "choice", "prompt": "Pick", "options": []})
            )

    def test_ask_human_rejects_duplicate_option_ids(self) -> None:
        with self.assertRaises(ValueError):
            ask_human().human_request_from_call(
                ToolCall(
                    "call_1",
                    "ask_human",
                    {
                        "kind": "choice",
                        "prompt": "Pick",
                        "options": [
                            {"id": "one", "label": "One"},
                            {"id": "one", "label": "Again"},
                        ],
                    },
                )
            )

    def test_ask_human_rejects_explicit_none_option_description(self) -> None:
        with self.assertRaises(ValueError):
            ask_human().human_request_from_call(
                ToolCall(
                    "call_1",
                    "ask_human",
                    {
                        "kind": "choice",
                        "prompt": "Pick",
                        "options": [{"id": "one", "label": "One", "description": None}],
                    },
                )
            )

    def test_ask_human_rejects_non_string_prompt(self) -> None:
        with self.assertRaises(ValueError):
            ask_human().human_request_from_call(
                ToolCall("call_1", "ask_human", {"kind": "input", "prompt": 1})
            )

    def test_ask_human_rejects_options_for_input(self) -> None:
        with self.assertRaises(ValueError):
            ask_human().human_request_from_call(
                ToolCall("call_1", "ask_human", {"kind": "input", "prompt": "Reply", "options": []})
            )
