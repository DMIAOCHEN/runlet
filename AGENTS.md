# Repository Instructions

These instructions apply to the entire repository.

## Project Identity

- Project name: Runlet.
- Python package name: `runlet`.
- GitHub repository: `https://github.com/DMIAOCHEN/runlet`.
- License: MIT.
- Current status: pre-alpha package skeleton.

## Product Direction

Runlet is a lightweight, provider-neutral Python agent runtime library. It is a
library, not a full application platform.

The core design goals are:

- Keep the runtime small and embeddable.
- Make context budgeting and compression mandatory before model calls.
- Provide flexible hooks around model calls, tool calls, state operations, and
  context compression.
- Emit structured events for observability.
- Keep model providers, storage, tracing backends, and application frameworks
  behind adapters.

## Non-Goals

Do not turn the core package into:

- A web application framework.
- A hosted agent platform.
- A worker queue.
- A UI or trace viewer.
- A multi-tenant control plane.
- A graph workflow engine for the first release.

## Design Drafts

Detailed internal design drafts, implementation planning notes, and private
brainstorming documents must go under:

```text
.runlet-design/
```

This directory is intentionally ignored by Git and must not be committed or
pushed to GitHub.

Public docs under `docs/` should stay high-level and suitable for open-source
readers. Do not put private planning notes, unfinished internal specs, or
conversation transcripts in public docs.

## Development Practices

- Prefer simple Python modules and explicit protocols over framework-heavy
  abstractions.
- Do not add provider SDK dependencies to core runtime modules.
- Use `apply_patch` for manual file edits.
- Keep generated caches such as `__pycache__/` out of commits.
- Current test command, when tests exist:

```bash
python3 -m unittest discover tests
```

## Git Notes

The default branch is `main`.

If commands need to write Git metadata in this environment, they may require
escalated permissions because `.git` can appear read-only inside the sandbox.

