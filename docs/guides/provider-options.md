# Provider Options

Provider-specific request options belong in `ModelRequest.options`, namespaced
by provider.

## Why

This keeps the runtime core provider-neutral while still allowing provider
adapters to expose vendor-specific controls.

## Current namespaces

- `openai`
- `openai_chat`

## OpenAI Responses example

```python
request = ModelRequest(
    messages=[Message.user("Summarize this briefly.")],
    options={
        "openai": {
            "extra_body": {
                "reasoning": {"effort": "medium"},
            },
        },
    },
)
```

## OpenAI Chat Completions example

```python
request = ModelRequest(
    messages=[Message.user("Summarize this briefly.")],
    options={
        "openai_chat": {
            "extra_body": {
                "store": False,
            },
            "temperature": 0.2,
        },
    },
)
```

## Recommendation

Keep constructor arguments for connection-level setup such as:

- `api_key`
- `base_url`
- custom SDK `client`

Keep request-level behavior in `ModelRequest.options`.
