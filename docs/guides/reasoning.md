# Reasoning

Runlet exposes provider-native reasoning when the underlying provider returns
it. Runlet does not synthesize reasoning text on its own.

## Non-streaming

Use `RunResult.reasoning` or `ModelResponse.reasoning`:

```python
result = await runtime.run_request(agent, request)
print(result.output)
print(result.reasoning)
```

## Streaming

Reasoning deltas are emitted as:

- `model.stream.reasoning_delta`

```python
async for event in runtime.stream_request(agent, request):
    if event.type == "model.stream.reasoning_delta":
        print(event.payload["delta"], end="")
```

## Important note

Whether reasoning is actually returned depends on the provider or gateway.
Some OpenAI-compatible gateways expose reasoning tokens in usage data without
returning readable reasoning text.

That behavior is upstream. Runlet only surfaces what the provider returns.
