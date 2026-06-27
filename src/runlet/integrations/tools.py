from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from runlet.core.messages import ToolCall, ToolResult


ToolHandler = Callable[[dict[str, Any], "ToolContext"], Awaitable[str]]


def _metadata_map() -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class ToolContext:
    run_id: str
    metadata: dict[str, Any] = field(default_factory=_metadata_map)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler


def _annotation_to_schema(annotation: Any) -> dict[str, str]:
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    return {"type": "string"}


def tool(func: Callable[..., Awaitable[str]]) -> ToolSpec:
    signature = inspect.signature(func)
    properties: dict[str, dict[str, str]] = {}
    required: list[str] = []
    for name, parameter in signature.parameters.items():
        required.append(name)
        properties[name] = _annotation_to_schema(parameter.annotation)

    async def handler(arguments: dict[str, Any], context: ToolContext) -> str:
        del context
        return await func(**arguments)

    return ToolSpec(
        name=func.__name__,
        description=(func.__doc__ or "").strip(),
        input_schema={"type": "object", "required": required, "properties": properties},
        handler=handler,
    )


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
    for name in schema.get("required", []):
        if name not in arguments:
            raise ValueError(f"Missing required tool argument: {name}")


async def execute_tool_call(
    call: ToolCall,
    tools: dict[str, ToolSpec],
    context: ToolContext,
) -> ToolResult:
    if call.name not in tools:
        raise ValueError(f"Tool not found: {call.name}")
    spec = tools[call.name]
    validate_arguments(spec.input_schema, call.arguments)
    content = await spec.handler(call.arguments, context)
    return ToolResult(call_id=call.id, name=call.name, content=str(content))
