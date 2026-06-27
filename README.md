# Runlet

Runlet is a tiny observable runtime for Python agents.

The project is a library, not an application framework. Its core direction is a
provider-neutral, async-first agent runtime with strict context budgeting,
structured observability, and flexible hooks around model and tool execution.

## Current Status

This repository is initialized with the package skeleton only. Runtime behavior
will be added after the design spec is finalized. The API is not stable yet.

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

## Development

Run the current test suite:

```bash
python3 -m unittest discover tests
```
