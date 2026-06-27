# Design Principles

Runlet is a small runtime library, not an agent application framework.

## Small Core

The core package should contain only the execution model and stable extension
contracts. Integrations with model providers, storage systems, tracing backends,
and application frameworks should live behind adapters.

## Provider Neutral

Runlet should not model its internals after a single LLM provider API. Provider
adapters translate between provider-specific request shapes and Runlet's own
messages, tool calls, usage data, and errors.

## Context Safe

Every model request must pass through context budgeting before it is sent.
Compression, trimming, and overflow behavior are runtime safety mechanisms, not
optional application helpers.

## Event First

The runtime should emit structured events for runs, steps, model calls, tool
calls, context changes, state changes, hooks, and failures. Logs and traces can
be derived from events.

## Hooks Are Behavior, Observers Are Visibility

Hooks may alter runtime behavior. Observers should only consume events. Keeping
these separate prevents tracing from becoming hidden control flow.

