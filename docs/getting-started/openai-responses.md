# OpenAI Responses

Use `OpenAIResponsesProvider` when you want the OpenAI Responses API shape.

For third-party OpenAI-compatible gateways, `OpenAIChatCompletionsProvider` is
often the safer first choice. Responses compatibility varies more across
gateways.

## Minimal example

```python
import asyncio
import os

from runlet import Agent, Runtime
from runlet.providers import OpenAIResponsesProvider


async def main() -> None:
    provider = OpenAIResponsesProvider(
        model="gpt-4.1-mini",
        api_key=os.environ["OPENAI_API_KEY"],
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

## Custom base URL

```python
provider = OpenAIResponsesProvider(
    model="gpt-4.1-mini",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://your-endpoint.example/v1",
)
```

## Request-level provider options

Provider-specific options live under `ModelRequest.options["openai"]`.

```python
from runlet import Message
from runlet.core.models import ModelRequest


request = ModelRequest(
    messages=[Message.user("Summarize this briefly.")],
    options={
        "openai": {
            "extra_body": {
                "reasoning": {"effort": "medium"},
            },
        },
    },
)

result = await Runtime().run_request(agent, request)
print(result.output)
print(result.reasoning)
```

## Notes

- `base_url` is supported
- request-level `extra_body` is supported
- streaming is supported
- runtime-managed tool calling is supported
- reasoning output depends on what the provider or gateway actually returns

## Related guides

- [Streaming](streaming.md)
- [Tool Calling](tool-calling.md)
- [Reasoning](../guides/reasoning.md)
