# Architecture

Runlet's first runtime design is an async, provider-neutral agent loop.

## Packages

- `runlet.core`: stable contracts such as agents, messages, run results, model
  request and response types, events, and error classes.
- `runlet.runtime`: runtime orchestration concerns such as the execution loop,
  context preparation, policies, and state storage.
- `runlet.integrations`: runtime extension surfaces such as tools and hooks.
- `runlet.testing`: fake providers and deterministic test helpers.

## Execution Flow

1. A caller invokes the runtime with an agent and input.
2. The runtime creates a run context and emits `run.started`.
3. The runtime builds a model request for the next step.
4. Hooks can modify the model request.
5. The context manager enforces the model context budget and compresses or
   truncates as configured.
6. The model provider is called.
7. Hooks can inspect or modify the model response.
8. Tool calls are validated, passed through hooks, executed, and recorded.
9. State updates and structured events are emitted.
10. The loop continues until a final result, policy stop, cancellation, or error.

## Non-Goals

Runlet core does not provide an HTTP server, UI, worker queue, multi-tenant
control plane, database migration layer, or graph workflow engine.
