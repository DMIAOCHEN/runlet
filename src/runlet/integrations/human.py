from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from runlet.core.human import HumanOption, HumanRequest
from runlet.core.messages import ToolCall
from runlet.integrations.tools import ToolContext, ToolSpec


async def _unreachable_human_input_handler(arguments: dict[str, Any], context: ToolContext) -> str:
    del arguments, context
    raise RuntimeError("Human input tools must be handled by the runtime.")


@dataclass(frozen=True)
class HumanInputToolSpec(ToolSpec):
    def human_request_from_call(self, call: ToolCall) -> HumanRequest:
        kind = call.arguments.get("kind")
        if kind not in {"choice", "input"}:
            raise ValueError("Human input kind must be 'choice' or 'input'.")

        prompt = call.arguments.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("Human input prompt must be a string.")

        if kind == "input":
            if "options" in call.arguments:
                raise ValueError("Input requests must not include options.")
            options: tuple[HumanOption, ...] = ()
        else:
            options = self._choice_options(call.arguments.get("options"))

        return HumanRequest(
            id=f"hitl_{uuid4().hex}",
            kind=kind,
            prompt=prompt,
            options=options,
        )

    @staticmethod
    def _choice_options(value: object) -> tuple[HumanOption, ...]:
        if not isinstance(value, list) or not value:
            raise ValueError("Choice requests require one or more options.")

        options: list[HumanOption] = []
        option_ids: set[str] = set()
        for value_option in value:
            if not isinstance(value_option, Mapping):
                raise ValueError("Choice options must be maps.")

            option_id = value_option.get("id")
            label = value_option.get("label")
            if not isinstance(option_id, str) or not option_id:
                raise ValueError("Choice option id must be a nonempty string.")
            if not isinstance(label, str) or not label:
                raise ValueError("Choice option label must be a nonempty string.")
            if option_id in option_ids:
                raise ValueError("Choice option ids must be unique.")

            description = value_option.get("description")
            if description is not None and not isinstance(description, str):
                raise ValueError("Choice option description must be a string.")

            option_ids.add(option_id)
            options.append(HumanOption(id=option_id, label=label, description=description))

        return tuple(options)


def ask_human() -> HumanInputToolSpec:
    return HumanInputToolSpec(
        name="ask_human",
        description="Request a structured choice or text input from a person.",
        input_schema={
            "type": "object",
            "required": ["kind", "prompt"],
            "properties": {
                "kind": {"type": "string"},
                "prompt": {"type": "string"},
                "options": {"type": "array"},
            },
        },
        handler=_unreachable_human_input_handler,
    )
