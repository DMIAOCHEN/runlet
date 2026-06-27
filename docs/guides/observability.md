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
- `run.completed`
- `policy.stopped`

## Example

```python
result = await runtime.run(agent, "用一句中文介绍 Runlet。")

for event in observer.events:
    print(event.type, event.payload)
```

## Why events matter

Events give applications a stable visibility layer without coupling tracing,
logging, or metrics to control flow.
