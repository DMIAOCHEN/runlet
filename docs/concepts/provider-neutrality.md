# Provider Neutrality

Runlet does not model its core runtime around a single provider API.

## The approach

Providers translate vendor-specific request and stream formats into Runlet's
own contracts:

- `Message`
- `ToolCall`
- `ModelRequest`
- `ModelResponse`
- `ProviderStreamEvent`

## Why it matters

This keeps the runtime loop stable even when:

- providers expose different SDKs
- request bodies differ
- stream event shapes differ
- tool call payloads differ

## Extension rule

Provider-specific behavior should usually be expressed in one of two places:

1. provider constructor arguments for connection-level setup
2. `ModelRequest.options[...]` for request-level settings

That separation keeps the runtime core small while still allowing adapters to
expose practical gateway features such as `base_url`, custom headers, or extra
request body fields.
