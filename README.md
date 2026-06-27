# Runlet

Runlet is a tiny observable runtime for Python agents.

The project is a library, not an application framework. Its core direction is a
provider-neutral, async-first agent runtime with strict context budgeting,
structured observability, and flexible hooks around model and tool execution.

## Current Status

This repository now contains an MVP runtime skeleton with core contracts,
events, tools, hooks, context budgeting, streaming, and in-memory state. The
API is not stable yet.

## Design Goals

- Keep the core runtime small and embeddable.
- Treat context budgeting and compression as mandatory runtime safety checks.
- Expose hooks before and after model calls, tool calls, state operations, and
  context compression.
- Emit structured events for runs, steps, model calls, tool calls, context
  changes, state changes, and failures.
- Stay provider-neutral: model SDKs integrate through adapters, not core
  dependencies.

## Non-Goals

Runlet core does not aim to provide:

- A web application framework.
- A hosted agent platform.
- A task queue or worker system.
- A UI or trace viewer.
- A multi-tenant control plane.
- A graph workflow engine in the first release.

## Project Documents

- [Architecture](docs/architecture.md)
- [Design Principles](docs/design-principles.md)
- [Comparison](docs/comparison.md)
- [Roadmap](docs/roadmap.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)

## Minimal Shape

```python
from runlet import Agent, Runtime, tool


@tool
async def lookup(order_id: str) -> str:
    return f"order {order_id}"


agent = Agent(
    name="support",
    instructions="Help users with orders.",
    model=my_model_provider,
    tools=(lookup,),
)

result = await Runtime().run(agent, "Where is order 123?")
```

## OpenAI Provider

Install the optional OpenAI dependency:

```bash
pip install "runlet[openai]"
```

Minimal example:

```python
from runlet import Agent, Runtime
from runlet.providers import OpenAIResponsesProvider


provider = OpenAIResponsesProvider(model="gpt-5.5")

agent = Agent(
    name="assistant",
    instructions="Be helpful.",
    model=provider,
)

result = await Runtime().run(agent, "Say hello in one sentence.")
```

Custom base URL:

```python
from runlet.providers import OpenAIResponsesProvider


provider = OpenAIResponsesProvider(
    model="gpt-5.5",
    base_url="https://your-endpoint.example/v1",
)
```

Provider-specific request options:

```python
from runlet.core import Message
from runlet.core.models import ModelRequest
from runlet.providers import OpenAIResponsesProvider


provider = OpenAIResponsesProvider(model="gpt-5.5")


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

response = await provider.complete(request)
```

Streaming text deltas:

```python
from runlet import Agent, Runtime
from runlet.providers import OpenAIResponsesProvider


provider = OpenAIResponsesProvider(model="gpt-5.5")
agent = Agent(
    name="assistant",
    instructions="Be helpful.",
    model=provider,
)

async for event in Runtime().stream(agent, "Explain recursion in one sentence."):
    if event.type == "model.stream.delta":
        print(event.payload["delta"], end="")
```

Current scope of the provider:

- `complete()` supported
- `capabilities()` supported
- text-delta `stream()` supported
- `base_url` supported
- `options["openai"]["extra_body"]` supported
- tool messages are not supported yet
- tool-call streaming is not supported yet

## Development

Run the current test suite:

```bash
PYTHONPATH=src python3 -m unittest discover tests
```
