# Runlet

Runlet is a small, provider-neutral Python agent runtime library.

It is built for applications that want explicit control over model calls, tool
execution, context budgeting, and structured observability without adopting a
large framework.

## Why Runlet

Runlet is a good fit when you want to:

- embed agent execution inside an existing Python application
- keep model providers behind adapters
- stream model output while still executing tools in the runtime loop
- enforce context preparation before model calls
- observe runs through structured events
- own conversation state and memory policy at the application layer

## What Runlet is not

Runlet is not trying to be:

- a hosted agent platform
- a web framework
- a graph workflow engine
- a UI or trace viewer
- a full memory framework

## Quickstart

Install Runlet:

```bash
pip install "runlet[openai]"
pip install python-dotenv
```

For Anthropic:

```bash
pip install "runlet[anthropic]"
```

Example `.env`:

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://your-endpoint.example/v1
OPENAI_MODEL=qwen-plus
```

Minimal example:

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

    result = await Runtime().run(agent, "Introduce Runlet in one sentence.")
    print(result.output)


asyncio.run(main())
```

## Start here

- [Installation](getting-started/installation.md)
- [First Agent](getting-started/first-agent.md)
- [OpenAI Chat Completions](getting-started/openai-chat-completions.md)
- [Anthropic Messages](getting-started/anthropic-messages.md)
- [Streaming](getting-started/streaming.md)
- [Tool Calling](getting-started/tool-calling.md)
- [Human In The Loop](guides/human-in-the-loop.md)
- [Full Example](getting-started/full-example.md)

## Core capabilities

- provider-neutral runtime loop
- async model execution
- streaming text output
- runtime-managed streaming tool execution
- request-level provider options
- structured runtime events
- lightweight state store primitives
