# Roadmap

This roadmap is intentionally small. Runlet should earn complexity through real
runtime needs, not start as a large framework.

## 0.1

- Define core data models and protocols.
- Implement the async runtime loop.
- Add model provider protocol and fake provider for tests.
- Add tool declaration and execution.
- Add structured events and in-memory observer.
- Add context budgeting with fail-closed overflow behavior.
- Add basic hook pipeline.

## 0.2

- Add context compression strategies.
- Add JSONL trace observer.
- Add OpenTelemetry observer behind optional dependencies.
- Add provider adapters outside the core runtime.
- Add richer tool policies and timeout handling.

## 0.3

- Add state store adapters.
- Add replay/debug helpers.
- Evaluate whether graph execution belongs in Runlet or in a separate package.

