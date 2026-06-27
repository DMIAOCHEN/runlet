# Architecture

Runlet's first runtime design is an async, provider-neutral agent loop.

## Modules

- `runlet.core`: domain objects such as agents, messages, tool calls, run input,
  run results, steps, and errors.
- `runlet.runtime`: the execution loop that coordinates model calls, tool calls,
  context preparation, hooks, state updates, and events.
- `runlet.models`: model provider protocols and provider capability metadata.
- `runlet.tools`: tool declarations, schemas, handlers, execution results, and
  tool errors.
- `runlet.context`: token budgeting, message reduction, compression, truncation,
  and overflow handling.
- `runlet.hooks`: behavior extension points around runs, steps, model calls,
  tool calls, context compression, and state operations.
- `runlet.events`: structured event definitions.
- `runlet.observers`: event consumers such as console, JSONL, in-memory, and
  OpenTelemetry adapters.
- `runlet.state`: state store protocols and default in-memory storage.
- `runlet.policies`: run limits, retry rules, timeout rules, context policy,
  hook policy, and tool policy.
- `runlet.testing`: fake providers, fake tools, event recorders, and deterministic
  helpers for tests.

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

