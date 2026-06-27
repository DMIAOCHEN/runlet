# First Agent

This is the shortest path to a working Runlet agent.

## What you will build

A single-turn agent that sends one user message to a model provider and prints
the final output.

## Example

```python
import asyncio
import os

from dotenv import load_dotenv

from runlet import Agent, Runtime
from runlet.providers import OpenAIChatCompletionsProvider


async def main() -> None:
    load_dotenv()

    provider = OpenAIChatCompletionsProvider(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    agent = Agent(
        name="assistant",
        instructions="Be concise and helpful.",
        model=provider,
    )

    runtime = Runtime()
    result = await runtime.run(agent, "Introduce Runlet in one sentence.")
    print(result.output)


asyncio.run(main())
```

Example `.env`:

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://your-endpoint.example/v1
OPENAI_MODEL=qwen-plus
```

## Key objects

- `Agent`: bundles instructions, model, tools, and hooks
- `Runtime`: executes the run loop
- `ModelProvider`: the provider adapter used by the runtime

## What happens internally

1. `Runtime.run()` builds a `ModelRequest`
2. context preparation runs before the model call
3. the provider executes the request
4. Runlet returns a `RunResult`

## Next step

- [OpenAI Chat Completions](openai-chat-completions.md)
- [Streaming](streaming.md)
- [Full Example](full-example.md)
