# Runlet

Runlet is a tiny observable runtime for Python agents.

The project is a library, not an application framework. Its core direction is a
provider-neutral, async-first agent runtime with strict context budgeting,
structured observability, and flexible hooks around model and tool execution.

## Current Status

This repository now contains an MVP runtime skeleton with core contracts,
events, tools, hooks, context budgeting, streaming, provider adapters, and
in-memory state. The API is not stable yet.

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

Streaming with tool execution:

```python
from runlet import Agent, Runtime, tool
from runlet.providers import OpenAIResponsesProvider


@tool
async def lookup_order(order_id: str) -> str:
    return f"order {order_id} shipped"


provider = OpenAIResponsesProvider(model="gpt-5.5")
agent = Agent(
    name="assistant",
    instructions="Use tools when needed.",
    model=provider,
    tools=(lookup_order,),
)

async for event in Runtime().stream(agent, "Check order 123 and tell me the result."):
    if event.type == "model.stream.delta":
        print(event.payload["delta"], end="")
```

When the provider emits a tool call during streaming, `Runtime.stream()` now
executes the tool, appends the tool result to the conversation, and continues
the next model round until the run completes.

Current scope of the provider:

- `complete()` supported
- `capabilities()` supported
- `stream()` supported
- text deltas supported
- streaming tool execution through `Runtime.stream()` supported
- `base_url` supported
- `options["openai"]["extra_body"]` supported
- provider-specific request options stay under `ModelRequest.options["openai"]`

Current streaming contract:

- providers can emit provider-neutral streaming step events internally
- `Runtime.stream()` handles multi-round tool execution loops
- OpenAI is the first provider implementation of this contract

## Development

Run the current test suite:

```bash
PYTHONPATH=src python3 -m unittest discover tests
```

## Releasing

Runlet publishes to PyPI from Git tags through GitHub Actions.

Release flow:

1. Update `[project].version` in `pyproject.toml`
2. Merge the release commit to `main`
3. Create a version tag such as `v0.1.0`
4. Push the tag to GitHub

The publish workflow will:

- verify the Git tag matches `pyproject.toml`
- run the test suite
- build `sdist` and `wheel`
- validate package metadata
- publish to PyPI through Trusted Publishing

Example:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Repository setup requirement:

- configure PyPI Trusted Publishing for this GitHub repository and the
  `.github/workflows/publish.yml` workflow
