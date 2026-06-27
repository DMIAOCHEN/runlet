# Installation

This guide shows the minimal installation paths for Runlet.

## Base package

Install the core package:

```bash
pip install runlet
```

## OpenAI-compatible providers

Install the optional OpenAI dependency:

```bash
pip install "runlet[openai]"
```

## Anthropic provider

Install the optional Anthropic dependency:

```bash
pip install "runlet[anthropic]"
```

If you want both:

```bash
pip install "runlet[openai,anthropic]"
```

If you want the latest pre-release explicitly:

```bash
pip install --pre "runlet[openai]"
```

## Python version

Runlet requires Python 3.10 or newer.

## Next step

Continue with [First Agent](first-agent.md).
