# Anthropic Messages

Use `AnthropicMessagesProvider` when you want Anthropic's official `messages`
API shape.

This provider is designed around the official API first. It also supports
connection-level `base_url` and request-level options so it can be used behind
compatible gateways when their behavior is close enough to Anthropic's message
format.

## Minimal example

```python
import asyncio
import os

from dotenv import load_dotenv

from runlet import Agent, Runtime
from runlet.providers import AnthropicMessagesProvider


async def main() -> None:
    load_dotenv()

    provider = AnthropicMessagesProvider(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
    )

    agent = Agent(
        name="assistant",
        instructions="Answer in concise English.",
        model=provider,
    )

    result = await Runtime().run(agent, "Introduce Runlet in one sentence.")
    print(result.output)


asyncio.run(main())
```

Example `.env`:

```dotenv
ANTHROPIC_API_KEY=your-api-key
ANTHROPIC_BASE_URL=https://your-endpoint.example
ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

## Custom base URL

Use `base_url` when you need to point the SDK at an Anthropic-compatible
gateway:

```python
provider = AnthropicMessagesProvider(
    model="claude-sonnet-4-20250514",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="https://your-endpoint.example",
)
```

## Request-level provider options

Provider-specific request options live under
`ModelRequest.options["anthropic"]`.

```python
from runlet import Message
from runlet.core.models import ModelRequest


request = ModelRequest(
    messages=[Message.user("Summarize this briefly.")],
    options={
        "anthropic": {
            "max_tokens": 512,
            "thinking": {
                "type": "enabled",
                "budget_tokens": 1024,
            },
            "metadata": {
                "user_id": "u_123",
            },
        },
    },
)

result = await Runtime().run_request(agent, request)
print(result.output)
print(result.reasoning)
```

## Supported behaviors

- non-streaming completion
- streaming text output
- runtime-managed tool calling
- Anthropic `tool_use` / `tool_result` message mapping
- request-level `max_tokens`
- request-level `temperature`
- request-level `thinking`
- request-level `metadata`
- request-level `stop_sequences`
- request-level `extra_headers`
- request-level `extra_body`

## Notes

- `max_tokens` defaults to `1024` if not explicitly provided
- thinking output is surfaced through `ModelResponse.reasoning`,
  `RunResult.reasoning`, and `model.stream.reasoning_delta`
- third-party gateways may diverge from Anthropic's official tool-calling
  format; the provider is optimized for the official API shape

## Related guides

- [Streaming](streaming.md)
- [Tool Calling](tool-calling.md)
- [Reasoning](../guides/reasoning.md)
- [Provider Options](../guides/provider-options.md)
