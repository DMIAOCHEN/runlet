# Streaming

Runlet supports streaming model output through `Runtime.stream()`.

## Basic streaming

```python
import asyncio

from runlet import Agent, Runtime
from runlet.providers import OpenAIChatCompletionsProvider


async def main() -> None:
    provider = OpenAIChatCompletionsProvider(
        model="gpt-4o-mini",
        api_key="your-api-key",
    )

    agent = Agent(
        name="assistant",
        instructions="Be concise and helpful.",
        model=provider,
    )

    async for event in Runtime().stream(agent, "用两句中文解释什么是流式输出。"):
        if event.type == "model.stream.delta":
            print(event.payload["delta"], end="")


asyncio.run(main())
```

## Streaming request entrypoint

If you need request-level options, use `Runtime.stream_request()`:

```python
from runlet import Message
from runlet.core.models import ModelRequest


request = ModelRequest(
    messages=[Message.user("Explain streaming briefly.")],
    options={"openai_chat": {"extra_body": {"store": False}}},
)

async for event in Runtime().stream_request(agent, request):
    if event.type == "model.stream.delta":
        print(event.payload["delta"], end="")
```

## Reasoning deltas

If the provider emits reasoning content, Runlet forwards it as:

- `model.stream.reasoning_delta`

```python
async for event in Runtime().stream_request(agent, request):
    if event.type == "model.stream.reasoning_delta":
        print(event.payload["delta"], end="")
```

## Next step

Continue with [Tool Calling](tool-calling.md).
