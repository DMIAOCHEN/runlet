# Architecture

Runlet's current architecture is a small, async, provider-neutral agent loop.

If you are new to the project, start with the getting-started guides in
`docs/getting-started/` first. This document is the maintainer-oriented view.

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
8. Tool calls are validated and passed through hooks. A call that requires approval, or a call to `ask_human()`, creates and saves a checkpoint before the runtime emits the interruption events.
9. The caller renders the `HumanRequest` and calls `Runtime.resume()` with a
   matching `HumanResponse`. An approved tool approval executes its handler and
   records its `ToolResult`; a rejected approval appends the fixed rejection
   result; only `ask_human()` choice and input values become serialized tool
   results for the original tool call. No response becomes a new user message.
10. Remaining tool calls execute, state updates and structured events are emitted, and the loop continues until a final result, policy stop, cancellation, or error.

## Streaming Flow

`Runtime.stream()` follows the same high-level loop as `Runtime.run()`, but each
assistant step is consumed as a stream.

1. The runtime prepares a `ModelRequest` for the current step.
2. The provider emits internal streaming step events.
3. Text deltas are forwarded as `model.stream.delta`.
4. Completed tool calls execute immediately unless they require approval or request human input.
5. For a human interruption, the runtime saves a checkpoint before yielding `human.requested` and `run.interrupted`; the caller resumes with a matching `HumanResponse`.
6. The caller resumes through `Runtime.resume()`, which returns a normal,
   non-streaming `RunResult`; it does not resume the streaming iterator.
7. The resumed run follows the same approval and human-input result semantics
   as the non-streaming execution flow.
8. If the assistant step completes without tool calls, the stream completes.

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

Runlet core does not provide a UI, HTTP API, auth, durable storage, worker queue, multi-tenant control plane, database migration layer, or graph workflow engine. A future `runlet-harness` may offer adapters for application-facing concerns, but it is not part of the core package.
