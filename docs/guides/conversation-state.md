# Conversation State

Runlet provides state store primitives, but it does not impose a built-in
conversation memory framework.

That boundary is intentional.

## What Runlet provides

Runlet includes:

- `StateScope`
- `StateStore`
- `InMemoryStateStore`

These are low-level storage primitives. They are useful for building
application-specific session behavior.

## What applications should own

Applications should decide:

- how to persist message history
- how many prior turns to keep
- when to summarize old turns
- whether tool results become durable memory
- how to partition state by user, thread, or session

## Recommended pattern

Use a session helper that:

1. loads prior messages from a state store
2. appends the new user message
3. calls `Runtime.run_request()` or `Runtime.stream_request()`
4. saves the updated conversation history

## Example shape

```python
from runlet import InMemoryStateStore, Message, StateScope
from runlet.core.models import ModelRequest


store = InMemoryStateStore()
scope = StateScope(kind="chat_session", key="session-1")

state = await store.load(scope)
history = state.get("messages", [])

messages = [
    *history,
    Message.user("那我刚才说了什么？"),
]

request = ModelRequest(messages=messages)
result = await runtime.run_request(agent, request)
```

## Why this stays out of `Runtime`

If `Runtime` owned conversation memory directly, the core would quickly absorb:

- session semantics
- message retention policy
- summarization policy
- retrieval policy
- user-level persistence policy

That would push Runlet toward a larger application framework, which is outside
its intended scope.

## Next step

See [Runtime Boundary](../concepts/runtime-boundary.md).
