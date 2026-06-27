# Core Models

Runlet keeps a small set of core types around runtime execution.

## High-frequency types

- `Agent`
- `Message`
- `ModelRequest`
- `ModelResponse`
- `RunResult`
- `RuntimeEvent`
- `ToolCall`
- `ToolResult`

## `Message`

`Message` is the normalized conversation unit used across providers.

Supported roles currently include:

- `system`
- `user`
- `assistant`
- `tool`

## `ModelRequest`

`ModelRequest` carries:

- `messages`
- `tools`
- `metadata`
- `options`

`options` is the extension point for provider-specific request settings.

## `RunResult`

`RunResult` is the final runtime output for non-streaming execution. It can
contain:

- final output text
- aggregated reasoning text
- usage information
- failure information

## `RuntimeEvent`

`RuntimeEvent` is the provider-neutral observability unit emitted by the
runtime.
