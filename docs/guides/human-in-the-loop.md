# Human In The Loop

Runlet can interrupt a run when a tool call requires human approval or when an agent calls the built-in `ask_human()` tool. The runtime saves a checkpoint before reporting the interruption. Your application renders the request, collects a response, and resumes the same run.

The core package provides the interruption contract and checkpoint protocol. It does not provide a UI, HTTP API, auth, or durable storage. A future `runlet-harness` project may provide adapters for those concerns, but it is not a dependency of Runlet core.

Checkpoint data contains the request messages, options, metadata, accumulated transcript, and pending calls, but never Python `ToolSpec` handlers. On resume, Runlet resolves tool specifications from the supplied `Agent`. A durable store therefore needs to serialize only checkpoint data, and the resuming process must supply an agent with the required tools. Durable adapters must require JSON-compatible metadata and options because checkpoint structures cannot enforce the arbitrary `Any` values accepted by the in-memory API.

## Approve a tool call

Create an approval-gated `ToolSpec` and give `Runtime` an in-memory checkpoint
store. The examples assume `provider` is a configured model adapter. In a
production application, replace `InMemoryCheckpointStore` with an
implementation of `CheckpointStore` backed by your own durable store.

```python
from runlet import Agent, HumanResponse, InMemoryCheckpointStore, Runtime, ToolSpec


async def refund(arguments, context):
    return f"Refunded order {arguments['order_id']}."


refund_tool = ToolSpec(
    name="refund",
    description="Refund an order.",
    input_schema={
        "type": "object",
        "required": ["order_id"],
        "properties": {"order_id": {"type": "string"}},
    },
    handler=refund,
    requires_approval=True,
)

runtime = Runtime(checkpoint_store=InMemoryCheckpointStore())
agent = Agent(
    name="support",
    instructions="Ask for approval before issuing refunds.",
    model=provider,
    tools=(refund_tool,),
)

result = await runtime.run(agent, "Refund order 123.")
if result.status == "interrupted":
    request = result.interruption
    assert request is not None
    assert result.checkpoint_id is not None

    # Render only the request object in the application UI.
    show_approval(request.prompt)

    result = await runtime.resume(
        agent,
        checkpoint_id=result.checkpoint_id,
        response=HumanResponse(request_id=request.id, action="approve"),
    )

print(result.output)
```

For an approval request, use `approve` or `reject`. `approve` can include `edited_arguments` when the application allows an operator to alter the tool arguments before execution.

## Ask a human to choose

Add `ask_human()` to an agent's tools when the model needs an application to collect input. The model calls it with `kind="choice"`, a prompt, and option objects containing `id`, `label`, and optional `description` values.

```python
from runlet import Agent, HumanResponse, InMemoryCheckpointStore, Runtime, ask_human


runtime = Runtime(checkpoint_store=InMemoryCheckpointStore())
agent = Agent(
    name="planner",
    instructions="When a plan is unclear, ask the user to choose an option.",
    model=provider,
    tools=(ask_human(),),
)

result = await runtime.run(agent, "Help me choose a plan.")
if result.status == "interrupted":
    request = result.interruption
    assert request is not None
    assert result.checkpoint_id is not None

    # Render only the request and its options.
    option_id = show_choice(request.prompt, request.options)

    result = await runtime.resume(
        agent,
        checkpoint_id=result.checkpoint_id,
        response=HumanResponse(
            request_id=request.id,
            action="select",
            value=option_id,
        ),
    )

print(result.output)
```

For `kind="input"`, render the prompt and resume with `HumanResponse(request_id=request.id, action="submit", value=text)`.

## Resume semantics

The response is not added as a new `Message.user`. For an approved tool
approval, Runlet executes the real handler and records its `ToolResult`. For a
rejected approval, it appends the fixed rejection tool result. Only
`ask_human()` choice and input values are serialized into a tool result tied to
the original `tool_call_id`.

After an interruption from `Runtime.stream()`, call `Runtime.resume()` with the
checkpoint and response. `resume()` returns a normal, non-streaming
`RunResult`; it does not resume the streaming iterator. Preserve both
`checkpoint_id` and the `HumanRequest.id` in the application session so a
response resumes the intended interruption.

`InMemoryCheckpointStore` is appropriate only while the runtime process stays alive. Implement `CheckpointStore` when the application needs durable checkpoints across restarts or processes. Core resume is not exactly-once across concurrent durable resume attempts: a durable adapter needs an atomic claim or lease, plus a resolved-result or idempotency policy for tool execution.
