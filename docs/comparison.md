# Comparison

Runlet is not trying to replace every agent framework. It is a small runtime
core for applications that need explicit control over execution, hooks, context
budgeting, and observability.

This page explains positioning. For actual usage, start with `README.md` and
the getting-started guides.

## OpenAI Agents SDK

The OpenAI Agents SDK has strong lifecycle and tracing ideas. Runlet should
borrow the clarity of runner-style execution and first-class tracing, but remain
provider-neutral at the core.

## Pydantic AI

Pydantic AI has useful type-safety and dependency-injection patterns. Runlet
should borrow strong schemas and explicit dependencies without making structured
output the central product identity.

## smolagents

smolagents shows that agent APIs can stay small. Runlet should keep the default
path simple while adding stronger runtime boundaries for hooks, context safety,
and observability.

## LangGraph

LangGraph is powerful for graph and workflow execution. Runlet should borrow the
importance of state and event streams, but should not start as a graph-first
workflow engine.
