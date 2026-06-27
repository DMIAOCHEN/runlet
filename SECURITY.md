# Security

Runlet is pre-alpha and should not be treated as production-hardened yet.

## Scope

Security-sensitive areas include:

- Tool execution and external resource access.
- Prompt, message, and tool-result logging.
- Context compression and artifact retention.
- Hook behavior that can modify model requests or tool calls.

## Reporting

Do not disclose suspected vulnerabilities in public issues. Use the repository's
private vulnerability reporting channel when available, or contact the maintainers
privately.

## Design Expectations

- Sensitive fields should be redacted before structured events are exported.
- Tool execution should be explicit and policy-controlled.
- Hooks that alter requests, responses, or tool calls should be observable.
- Context overflow should fail closed unless a configured compression policy can
  bring the request under budget.

