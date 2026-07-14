# Runlet

[![PyPI version](https://img.shields.io/pypi/v/runlet.svg)](https://pypi.org/project/runlet/)
[![Python versions](https://img.shields.io/pypi/pyversions/runlet.svg)](https://pypi.org/project/runlet/)
[![CI](https://github.com/DMIAOCHEN/runlet/actions/workflows/ci.yml/badge.svg)](https://github.com/DMIAOCHEN/runlet/actions/workflows/ci.yml)
[![Docs](https://github.com/DMIAOCHEN/runlet/actions/workflows/docs.yml/badge.svg)](https://github.com/DMIAOCHEN/runlet/actions/workflows/docs.yml)

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

Install the Anthropic optional dependency:

```bash
pip install "runlet[anthropic]"
```

If you prefer `.env` based local development:

```bash
pip install python-dotenv
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
- [Anthropic Messages](docs/getting-started/anthropic-messages.md)
- [OpenAI Responses](docs/getting-started/openai-responses.md)
- [Streaming](docs/getting-started/streaming.md)
- [Tool Calling](docs/getting-started/tool-calling.md)
- [Full Example](docs/getting-started/full-example.md)

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

Runlet currently includes three built-in providers:

- `OpenAIChatCompletionsProvider`
- `OpenAIResponsesProvider`
- `AnthropicMessagesProvider`

For most third-party OpenAI-compatible gateways, start with
`OpenAIChatCompletionsProvider`. It is usually the more portable option.

For Anthropic's official API, use `AnthropicMessagesProvider`.

## Project status

Runlet is currently in beta. The API may still evolve before a stable 0.x release.

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
3. Create a tag such as `v0.2.0b2`
4. Push the tag

```bash
git tag v0.2.0b2
git push origin v0.2.0b2
```
