# Full Example

This example shows the main Runlet path in one place:

- load config from `.env`
- build an OpenAI-compatible provider
- run a normal request
- stream text output
- execute tools inside the runtime loop
- persist short conversation history in application state

## Example `.env`

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://your-endpoint.example/v1
OPENAI_MODEL=qwen-plus
```

## Example

```python
import asyncio
import os

from dotenv import load_dotenv

from runlet import Agent, InMemoryStateStore, Message, ModelRequest, Runtime, StateScope, tool
from runlet.providers import OpenAIChatCompletionsProvider


@tool
async def lookup_order(order_id: str) -> str:
    return f"order {order_id}: shipped, warehouse=hangzhou, eta=tomorrow"


def load_history_items(items: list[dict[str, str]]) -> list[Message]:
    messages: list[Message] = []
    for item in items:
        role = item["role"]
        text = item["text"]
        if role == "assistant":
            messages.append(Message.assistant(text))
        else:
            messages.append(Message.user(text))
    return messages


async def main() -> None:
    load_dotenv()

    provider = OpenAIChatCompletionsProvider(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    runtime = Runtime()
    agent = Agent(
        name="support",
        instructions=(
            "You are a concise Chinese support assistant. "
            "Use tools when the user asks for live order data."
        ),
        model=provider,
        tools=(lookup_order,),
    )

    result = await runtime.run(agent, "Introduce Runlet in one sentence.")
    print("run() ->", result.output)

    async for event in runtime.stream(agent, "Explain streaming output in two sentences."):
        if event.type == "model.stream.delta":
            print(event.payload["delta"], end="")
    print()

    tool_result = await runtime.run(agent, "Check order 12345 and briefly tell me its current status.")
    print("tool run() ->", tool_result.output)

    store = InMemoryStateStore()
    scope = StateScope(kind="chat_session", key="demo")

    state = await store.load(scope)
    history_items = state.get("messages", [])
    if not isinstance(history_items, list):
        history_items = []

    history = load_history_items(history_items)
    request = ModelRequest(
        messages=[
            *history,
            Message.user("Remember that my name is Miko and I am building Runlet."),
        ],
        options={"openai_chat": {"extra_body": {"store": False}}},
    )
    remember = await runtime.run_request(agent, request)
    print("memory turn 1 ->", remember.output)

    await store.save(
        scope,
        {
            "messages": [
                *history_items,
                {"role": "user", "text": "Remember that my name is Miko and I am building Runlet."},
                {"role": "assistant", "text": remember.output},
            ]
        },
    )

    state = await store.load(scope)
    next_history_items = state.get("messages", [])
    if not isinstance(next_history_items, list):
        next_history_items = []

    follow_up = await runtime.run_request(
        agent,
        ModelRequest(
            messages=[
                *load_history_items(next_history_items),
                Message.user("What did I just say I am building?"),
            ],
            options={"openai_chat": {"extra_body": {"store": False}}},
        ),
    )
    print("memory turn 2 ->", follow_up.output)


asyncio.run(main())
```

## What this demonstrates

- `Runtime.run()` for simple one-shot calls
- `Runtime.stream()` for text streaming
- runtime-managed tool execution
- `Runtime.run_request()` for request-level provider options
- application-level state management on top of `InMemoryStateStore`

## Next step

- [Conversation State](../guides/conversation-state.md)
- [Provider Options](../guides/provider-options.md)
