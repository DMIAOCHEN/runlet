# Runlet

Runlet is a small, provider-neutral Python agent runtime library.

It is designed for applications that want explicit control over model calls,
tool execution, context budgeting, and structured observability without
adopting a large framework.

## When to use Runlet

Runlet is a good fit when you want to:

- embed agent execution inside an existing Python application
- keep model providers behind adapters
- enforce context preparation before model calls
- stream model output while still executing tools inside the runtime loop
- observe runs through structured events
- build your own application-level conversation, memory, or state policies

## When not to use Runlet

Runlet is not trying to be:

- a hosted agent platform
- a web framework
- a graph workflow engine
- a UI or trace viewer
- a full memory framework

If you want a batteries-included platform with orchestration, persistence, and
application scaffolding built in, Runlet is intentionally narrower than that.

## Quickstart

Install Runlet:

```bash
pip install runlet
```

Install the OpenAI optional dependency:

```bash
pip install "runlet[openai]"
```

Minimal example:

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

    result = await Runtime().run(agent, "用一句中文介绍 Runlet。")
    print(result.output)


asyncio.run(main())
```

## Core capabilities

- provider-neutral runtime loop
- async model execution
- streaming text output
- runtime-managed streaming tool execution
- request-level provider options
- structured runtime events
- hook points around model and tool execution
- lightweight state store primitives

## Documentation

Start here:

- [Installation](docs/getting-started/installation.md)
- [First Agent](docs/getting-started/first-agent.md)
- [OpenAI Chat Completions](docs/getting-started/openai-chat-completions.md)
- [OpenAI Responses](docs/getting-started/openai-responses.md)
- [Streaming](docs/getting-started/streaming.md)
- [Tool Calling](docs/getting-started/tool-calling.md)

Guides:

- [Conversation State](docs/guides/conversation-state.md)
- [Provider Options](docs/guides/provider-options.md)
- [Reasoning](docs/guides/reasoning.md)
- [Observability](docs/guides/observability.md)

Concepts:

- [Runtime Boundary](docs/concepts/runtime-boundary.md)
- [Core Models](docs/concepts/core-models.md)
- [Provider Neutrality](docs/concepts/provider-neutrality.md)

Project background:

- [Architecture](docs/architecture.md)
- [Design Principles](docs/design-principles.md)
- [Comparison](docs/comparison.md)
- [Roadmap](docs/roadmap.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)

## Providers

Runlet currently includes two OpenAI-compatible providers:

- `OpenAIChatCompletionsProvider`
- `OpenAIResponsesProvider`

For most third-party OpenAI-compatible gateways, start with
`OpenAIChatCompletionsProvider`. It is usually the more portable option.

## Project status

Runlet is currently pre-alpha. The API is not stable yet.

## Development

Run the test suite:

```bash
PYTHONPATH=src python3 -m unittest discover tests
```

Run type checking:

```bash
pyright
```

## Release

Runlet publishes to PyPI from Git tags through GitHub Actions.

Typical release flow:

1. Update the version in `pyproject.toml`
2. Merge to `main`
3. Create a tag such as `v0.2.0a3`
4. Push the tag

```bash
git tag v0.2.0a3
git push origin v0.2.0a3
```
