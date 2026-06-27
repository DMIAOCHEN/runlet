# Tool Calling

Runlet can execute tools during both non-streaming and streaming runs.

## Define a tool

```python
from runlet import tool


@tool
async def lookup_order(order_id: str) -> str:
    return f"order {order_id}: shipped"
```

## Attach the tool to an agent

```python
from runlet import Agent


agent = Agent(
    name="support",
    instructions="Use tools when needed before answering.",
    model=provider,
    tools=(lookup_order,),
)
```

## Non-streaming execution

```python
result = await Runtime().run(agent, "Check order 12345.")
print(result.output)
```

If the model returns tool calls, Runlet will:

1. execute the tool
2. append the tool result to the conversation
3. continue the next model round
4. stop when the model returns a final answer

## Streaming execution

The same loop works in `Runtime.stream()` and `Runtime.stream_request()`.

```python
async for event in Runtime().stream(agent, "Check order 12345 and tell me its current status."):
    if event.type == "model.stream.delta":
        print(event.payload["delta"], end="")
```

When a streamed model step produces a tool call, the runtime executes the tool
immediately and starts the next streamed model step.

## Multi-tool flows

If the model decides to call multiple tools across multiple rounds, the runtime
continues that loop until it reaches a final assistant message or the run
policy stops it.

## Related guides

- [Conversation State](../guides/conversation-state.md)
- [Observability](../guides/observability.md)
