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

## Streaming Flow

`Runtime.stream()` follows the same high-level loop as `Runtime.run()`, but each
assistant step is consumed as a stream.

1. The runtime prepares a `ModelRequest` for the current step.
2. The provider emits internal streaming step events.
3. Text deltas are forwarded as `model.stream.delta`.
4. Completed tool calls are executed immediately by the runtime.
5. Tool results are appended to the message list.
6. If a tool was executed, the runtime starts the next streamed model step.
7. If the assistant step completes without tool calls, the run completes.

This keeps streaming tool execution in the runtime loop rather than pushing it
into provider-specific code.

## Provider Streaming Boundary

Providers are responsible for translating vendor-specific stream events into a
Runlet internal event layer. The current event kinds include:

- `text_delta`
- `tool_call_delta`
- `tool_call_completed`
- `message_completed`
- `usage`
- `completed`

This boundary keeps `runlet.runtime` provider-neutral while still allowing each
provider adapter to map its own SDK semantics into the runtime loop.

## Non-Goals

Runlet core does not provide an HTTP server, UI, worker queue, multi-tenant
control plane, database migration layer, or graph workflow engine.
