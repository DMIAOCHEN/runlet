# Contributing

Runlet is early-stage. Contributions should preserve the project direction:
small core, explicit extension points, strict context safety, and structured
observability.

## Development

Use Python 3.10 or newer.

```bash
python3 -m unittest discover tests
```

## Contribution Guidelines

- Keep runtime behavior provider-neutral.
- Prefer protocols and small data objects over framework-heavy abstractions.
- Do not add application-layer features such as web servers, task queues, UI, or
  multi-tenant management to the core package.
- Add tests for runtime behavior once runtime modules exist.
- Document new extension points before exposing them as public API.

## Pull Requests

Before opening a pull request, include:

- What changed and why.
- The runtime API or behavior impact.
- The observability impact, if any.
- The context-budgeting impact, if any.
- The test command you ran.

