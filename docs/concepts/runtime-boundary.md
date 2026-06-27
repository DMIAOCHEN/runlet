# Runtime Boundary

Runlet is a runtime library, not an application platform.

## What belongs in the runtime

The runtime should own:

- the execution loop
- context preparation before model calls
- tool execution inside the loop
- hook invocation
- structured event emission
- run policies and stop conditions

## What should stay outside

The runtime should not own:

- product-specific session logic
- durable user memory policy
- retrieval pipelines
- HTTP APIs
- background job orchestration
- multi-tenant control planes

## Practical rule

If a behavior is application-specific and different teams would reasonably want
to implement it differently, it should usually live outside `Runtime`.

That is why Runlet exposes low-level state primitives and request entrypoints
instead of a built-in conversation memory system.
