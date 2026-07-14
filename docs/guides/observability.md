# Observability

Runlet emits structured runtime events through an event sink.

## Basic observer

Use `InMemoryObserver` to capture emitted events:

```python
from runlet import InMemoryObserver, Runtime


observer = InMemoryObserver()
runtime = Runtime(event_sink=observer)
```

## Common event types

- `run.started`
- `context.budget_checked`
- `model.requested`
- `model.completed`
- `model.stream.started`
- `model.stream.delta`
- `model.stream.reasoning_delta`
- `model.stream.completed`
- `tool.started`
- `tool.completed`
- `human.requested`
- `human.responded`
- `human.response_rejected`
- `run.interrupted`
- `run.resumed`
- `run.completed`
- `policy.stopped`

## Human-in-the-loop events

When a tool requires approval or calls `ask_human()`, Runlet saves a checkpoint and then emits `human.requested` followed by `run.interrupted`. When the application resumes the checkpoint, it emits `human.responded` and `run.resumed`. Invalid or stale responses emit `human.response_rejected`.

Human event payloads identify the request, checkpoint, request kind, and where applicable the response action. They intentionally omit prompts, choices, and submitted values so applications can observe the flow without exposing human input content in routine event sinks.

## Example

```python
result = await runtime.run(agent, "用一句中文介绍 Runlet。")

for event in observer.events:
    print(event.type, event.payload)
```

## Why events matter

Events give applications a stable visibility layer without coupling tracing,
logging, or metrics to control flow.
