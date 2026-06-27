# OpenAI Chat Completions

Use `OpenAIChatCompletionsProvider` when you want the OpenAI chat completions
API shape. For many third-party OpenAI-compatible gateways, this is the best
starting point.

## Minimal example

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
        instructions="Answer in concise Chinese.",
        model=provider,
    )

    result = await Runtime().run(agent, "Introduce Runlet in one sentence.")
    print(result.output)


asyncio.run(main())
```

Example `.env`:

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://your-endpoint.example/v1
OPENAI_MODEL=qwen-plus
```

## Custom base URL

Use `base_url` when calling an OpenAI-compatible gateway:

```python
provider = OpenAIChatCompletionsProvider(
    model="qwen-plus",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
```

## Request-level provider options

Provider-specific request options live under `ModelRequest.options["openai_chat"]`.

```python
from runlet import Message
from runlet.core.models import ModelRequest


request = ModelRequest(
    messages=[Message.user("Summarize this briefly.")],
    options={
        "openai_chat": {
            "extra_body": {
                "store": False,
            },
            "temperature": 0.2,
        },
    },
)

result = await Runtime().run_request(agent, request)
print(result.output)
```

## Supported behaviors

- non-streaming completion
- streaming text output
- runtime-managed tool calling
- request-level `extra_body`
- request-level `extra_headers`
- request-level `temperature`
- request-level `max_tokens`
- provider-native reasoning output when the gateway returns it

## Related guides

- [Streaming](streaming.md)
- [Tool Calling](tool-calling.md)
- [Full Example](full-example.md)
- [Provider Options](../guides/provider-options.md)
