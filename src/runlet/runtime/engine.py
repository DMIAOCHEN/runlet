from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

from runlet.core.agent import Agent
from runlet.core.events import EventSink, InMemoryObserver, RuntimeEvent
from runlet.core.messages import Message, ToolCall
from runlet.core.models import ModelRequest, ModelStreamEvent, ProviderStreamEvent
from runlet.core.runs import RunResult
from runlet.integrations.hooks import HookRunner
from runlet.integrations.tools import ToolContext, ToolSpec, execute_tool_call
from runlet.runtime.context import ContextManager
from runlet.runtime.policies import RunPolicy


class Runtime:
    def __init__(
        self,
        event_sink: EventSink | None = None,
        context_manager: ContextManager | None = None,
        policy: RunPolicy | None = None,
    ) -> None:
        self.event_sink = event_sink or InMemoryObserver()
        self.context_manager = context_manager or ContextManager()
        self.policy = policy or RunPolicy()

    async def run(self, agent: Agent, input: str) -> RunResult:
        return await self.run_request(
            agent,
            ModelRequest(messages=[Message.system(agent.instructions), Message.user(input)]),
            input=input,
        )

    async def run_request(
        self,
        agent: Agent,
        request: ModelRequest,
        *,
        input: str | None = None,
    ) -> RunResult:
        run_id = f"run_{uuid4().hex}"
        hook_runner = HookRunner(agent.hooks)
        await self.event_sink.emit(
            RuntimeEvent(
                type="run.started",
                run_id=run_id,
                agent_name=agent.name,
                payload={"input": input if input is not None else self._request_input_text(request)},
            )
        )

        request = self._merge_request_defaults(agent, request)
        messages = list(request.messages)
        tools = {tool_spec.name: tool_spec for tool_spec in request.tools if isinstance(tool_spec, ToolSpec)}
        collected_reasoning: list[str] = []

        for _ in range(self.policy.max_steps):
            step_request = self._step_request_from_state(request, messages, tools)
            step_request = await hook_runner.before_model_request(step_request, None)
            capabilities = await agent.model.capabilities()
            prepared = await self.context_manager.prepare(step_request, capabilities)
            await self.event_sink.emit(
                RuntimeEvent(
                    type="context.budget_checked",
                    run_id=run_id,
                    agent_name=agent.name,
                    payload={"input_tokens": prepared.estimate.input_tokens},
                )
            )
            await self.event_sink.emit(RuntimeEvent(type="model.requested", run_id=run_id, agent_name=agent.name))
            response = await agent.model.complete(prepared.request)
            response = await hook_runner.after_model_response(response, None)
            if response.reasoning:
                collected_reasoning.append(response.reasoning)
            await self.event_sink.emit(
                RuntimeEvent(
                    type="model.completed",
                    run_id=run_id,
                    agent_name=agent.name,
                    payload={"usage": response.usage.total_tokens},
                )
            )

            if response.tool_calls:
                messages.append(
                    Message.assistant(
                        response.message.text,
                        metadata={
                            "tool_calls": [
                                {"id": call.id, "name": call.name, "arguments": dict(call.arguments)}
                                for call in response.tool_calls
                            ]
                        },
                    )
                )
                for call in response.tool_calls:
                    call = await hook_runner.before_tool_call(call, None)
                    await self.event_sink.emit(
                        RuntimeEvent(
                            type="tool.started",
                            run_id=run_id,
                            agent_name=agent.name,
                            payload={"name": call.name},
                        )
                    )
                    tool_result = await execute_tool_call(call, tools, ToolContext(run_id=run_id))
                    tool_result = await hook_runner.after_tool_result(tool_result, None)
                    await self.event_sink.emit(
                        RuntimeEvent(
                            type="tool.completed",
                            run_id=run_id,
                            agent_name=agent.name,
                            payload={"name": tool_result.name},
                        )
                    )
                    messages.append(
                        Message.tool(
                            tool_result.content,
                            metadata={"tool_call_id": tool_result.call_id, "name": tool_result.name},
                        )
                    )
                continue

            run_result = RunResult.completed(
                run_id=run_id,
                output=response.message.text,
                reasoning="".join(collected_reasoning),
                usage=response.usage,
            )
            await self.event_sink.emit(
                RuntimeEvent(
                    type="run.completed",
                    run_id=run_id,
                    agent_name=agent.name,
                    payload={"output": run_result.output, "reasoning": run_result.reasoning},
                )
            )
            return run_result

        run_result = RunResult.failed(
            run_id=run_id,
            error="Maximum steps exceeded",
            reasoning="".join(collected_reasoning),
        )
        await self.event_sink.emit(
            RuntimeEvent(
                type="policy.stopped",
                run_id=run_id,
                agent_name=agent.name,
                payload={"reason": "max_steps"},
            )
        )
        return run_result

    async def stream(self, agent: Agent, input: str):
        async for event in self.stream_request(
            agent,
            ModelRequest(messages=[Message.system(agent.instructions), Message.user(input)]),
            input=input,
        ):
            yield event

    async def stream_request(
        self,
        agent: Agent,
        request: ModelRequest,
        *,
        input: str | None = None,
    ):
        run_id = f"run_{uuid4().hex}"
        hook_runner = HookRunner(agent.hooks)
        await self.event_sink.emit(
            RuntimeEvent(
                type="run.started",
                run_id=run_id,
                agent_name=agent.name,
                payload={"input": input if input is not None else self._request_input_text(request)},
            )
        )

        request = self._merge_request_defaults(agent, request)
        messages = list(request.messages)
        tools = {tool_spec.name: tool_spec for tool_spec in request.tools if isinstance(tool_spec, ToolSpec)}
        all_text_deltas: list[str] = []
        all_reasoning_deltas: list[str] = []

        for _ in range(self.policy.max_steps):
            step_request = self._step_request_from_state(request, messages, tools)
            step_request = await hook_runner.before_model_request(step_request, None)
            capabilities = await agent.model.capabilities()
            prepared = await self.context_manager.prepare(step_request, capabilities)
            await self.event_sink.emit(
                RuntimeEvent(
                    type="context.budget_checked",
                    run_id=run_id,
                    agent_name=agent.name,
                    payload={"input_tokens": prepared.estimate.input_tokens},
                )
            )

            await self.event_sink.emit(
                RuntimeEvent(
                    type="model.stream.started",
                    run_id=run_id,
                    agent_name=agent.name,
                )
            )

            saw_message_completed = False
            executed_tool = False
            collected_deltas: list[str] = []
            collected_reasoning_deltas: list[str] = []
            completed_tool_calls: list[dict[str, object]] = []
            pending_tool_messages: list[Message] = []
            async for step_event in self._iter_provider_stream_events(agent.model.stream(prepared.request)):
                if step_event.kind == "text_delta" and step_event.delta:
                    collected_deltas.append(step_event.delta)
                    all_text_deltas.append(step_event.delta)
                    event = RuntimeEvent(
                        type="model.stream.delta",
                        run_id=run_id,
                        agent_name=agent.name,
                        payload={"delta": step_event.delta},
                    )
                    await self.event_sink.emit(event)
                    yield event
                    continue

                if step_event.kind == "reasoning_delta" and step_event.reasoning:
                    collected_reasoning_deltas.append(step_event.reasoning)
                    all_reasoning_deltas.append(step_event.reasoning)
                    event = RuntimeEvent(
                        type="model.stream.reasoning_delta",
                        run_id=run_id,
                        agent_name=agent.name,
                        payload={"delta": step_event.reasoning},
                    )
                    await self.event_sink.emit(event)
                    yield event
                    continue

                if step_event.kind == "tool_call_completed" and step_event.call_id and step_event.name:
                    call = ToolCall(id=step_event.call_id, name=step_event.name, arguments=step_event.arguments)
                    call = await hook_runner.before_tool_call(call, None)
                    completed_tool_calls.append(
                        {"id": call.id, "name": call.name, "arguments": dict(call.arguments)}
                    )
                    await self.event_sink.emit(
                        RuntimeEvent(
                            type="tool.started",
                            run_id=run_id,
                            agent_name=agent.name,
                            payload={"name": call.name},
                        )
                    )
                    tool_result = await execute_tool_call(call, tools, ToolContext(run_id=run_id))
                    tool_result = await hook_runner.after_tool_result(tool_result, None)
                    await self.event_sink.emit(
                        RuntimeEvent(
                            type="tool.completed",
                            run_id=run_id,
                            agent_name=agent.name,
                            payload={"name": tool_result.name},
                        )
                    )
                    pending_tool_messages.append(
                        Message.tool(
                            tool_result.content,
                            metadata={"tool_call_id": tool_result.call_id, "name": tool_result.name},
                        )
                    )
                    executed_tool = True
                    continue

                if step_event.kind == "message_completed":
                    saw_message_completed = True

            await self.event_sink.emit(
                RuntimeEvent(
                    type="model.stream.completed",
                    run_id=run_id,
                    agent_name=agent.name,
                )
            )

            if executed_tool:
                messages.append(
                    Message.assistant(
                        "".join(collected_deltas),
                        metadata={
                            "tool_calls": completed_tool_calls,
                            "reasoning": "".join(collected_reasoning_deltas),
                        },
                    )
                )
                messages.extend(pending_tool_messages)
                continue

            if saw_message_completed:
                await self.event_sink.emit(
                    RuntimeEvent(
                        type="run.completed",
                        run_id=run_id,
                        agent_name=agent.name,
                        payload={"output": "".join(all_text_deltas), "reasoning": "".join(all_reasoning_deltas)},
                    )
                )
                return

        await self.event_sink.emit(
            RuntimeEvent(
                type="policy.stopped",
                run_id=run_id,
                agent_name=agent.name,
                payload={"reason": "max_steps"},
            )
        )

    async def _iter_provider_stream_events(
        self,
        stream: AsyncIterator[ProviderStreamEvent | ModelStreamEvent],
    ) -> AsyncIterator[ProviderStreamEvent]:
        async for chunk in stream:
            if isinstance(chunk, ProviderStreamEvent):
                yield chunk
                continue

            if chunk.delta:
                yield ProviderStreamEvent.text_delta(chunk.delta, raw=chunk.raw)
            if getattr(chunk, "reasoning", ""):
                yield ProviderStreamEvent.reasoning_delta(getattr(chunk, "reasoning"), raw=chunk.raw)
            if chunk.tool_call is not None:
                yield ProviderStreamEvent.tool_call_completed(
                    call_id=chunk.tool_call.id,
                    name=chunk.tool_call.name,
                    arguments=chunk.tool_call.arguments,
                    raw=chunk.raw,
                )
            if chunk.usage is not None:
                yield ProviderStreamEvent.usage_event(chunk.usage, raw=chunk.raw)
            if chunk.final:
                yield ProviderStreamEvent.message_completed(raw=chunk.raw)
                yield ProviderStreamEvent.completed(raw=chunk.raw)

    def _merge_request_defaults(self, agent: Agent, request: ModelRequest) -> ModelRequest:
        if request.messages and request.messages[0].role == "system":
            messages = list(request.messages)
        else:
            messages = [Message.system(agent.instructions), *request.messages]

        tools = list(request.tools) if request.tools else [tool for tool in agent.tools if isinstance(tool, ToolSpec)]
        return ModelRequest(
            messages=messages,
            tools=tools,
            metadata=dict(request.metadata),
            options=dict(request.options),
        )

    def _step_request_from_state(
        self,
        base_request: ModelRequest,
        messages: list[Message],
        tools: dict[str, ToolSpec],
    ) -> ModelRequest:
        return ModelRequest(
            messages=list(messages),
            tools=list(tools.values()),
            metadata=dict(base_request.metadata),
            options=dict(base_request.options),
        )

    def _request_input_text(self, request: ModelRequest) -> str:
        for message in reversed(request.messages):
            if message.role == "user":
                return message.text
        return ""
