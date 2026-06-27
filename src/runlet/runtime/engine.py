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
        run_id = f"run_{uuid4().hex}"
        hook_runner = HookRunner(agent.hooks)
        await self.event_sink.emit(
            RuntimeEvent(
                type="run.started",
                run_id=run_id,
                agent_name=agent.name,
                payload={"input": input},
            )
        )

        messages = [Message.system(agent.instructions), Message.user(input)]
        tools = {tool_spec.name: tool_spec for tool_spec in agent.tools if isinstance(tool_spec, ToolSpec)}

        for _ in range(self.policy.max_steps):
            request = ModelRequest(messages=list(messages), tools=list(tools.values()))
            request = await hook_runner.before_model_request(request, None)
            capabilities = await agent.model.capabilities()
            prepared = await self.context_manager.prepare(request, capabilities)
            await self.event_sink.emit(
                RuntimeEvent(
                    type="context.budget_checked",
                    run_id=run_id,
                    agent_name=agent.name,
                    payload={"input_tokens": prepared.estimate.input_tokens},
                )
            )
            request = prepared.request
            await self.event_sink.emit(RuntimeEvent(type="model.requested", run_id=run_id, agent_name=agent.name))
            response = await agent.model.complete(request)
            response = await hook_runner.after_model_response(response, None)
            await self.event_sink.emit(
                RuntimeEvent(
                    type="model.completed",
                    run_id=run_id,
                    agent_name=agent.name,
                    payload={"usage": response.usage.total_tokens},
                )
            )

            if response.tool_calls:
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
                    result = await execute_tool_call(call, tools, ToolContext(run_id=run_id))
                    result = await hook_runner.after_tool_result(result, None)
                    await self.event_sink.emit(
                        RuntimeEvent(
                            type="tool.completed",
                            run_id=run_id,
                            agent_name=agent.name,
                            payload={"name": result.name},
                        )
                    )
                    messages.append(
                        Message.tool(
                            result.content,
                            metadata={"tool_call_id": result.call_id, "name": result.name},
                        )
                    )
                continue

            result = RunResult.completed(run_id=run_id, output=response.message.text, usage=response.usage)
            await self.event_sink.emit(
                RuntimeEvent(
                    type="run.completed",
                    run_id=run_id,
                    agent_name=agent.name,
                    payload={"output": result.output},
                )
            )
            return result

        result = RunResult.failed(run_id=run_id, error="Maximum steps exceeded")
        await self.event_sink.emit(
            RuntimeEvent(
                type="policy.stopped",
                run_id=run_id,
                agent_name=agent.name,
                payload={"reason": "max_steps"},
            )
        )
        return result

    async def stream(self, agent: Agent, input: str):
        run_id = f"run_{uuid4().hex}"
        hook_runner = HookRunner(agent.hooks)
        messages = [Message.system(agent.instructions), Message.user(input)]
        tools = {tool_spec.name: tool_spec for tool_spec in agent.tools if isinstance(tool_spec, ToolSpec)}
        await self.event_sink.emit(
            RuntimeEvent(
                type="run.started",
                run_id=run_id,
                agent_name=agent.name,
                payload={"input": input},
            )
        )

        for _ in range(self.policy.max_steps):
            request = ModelRequest(messages=list(messages), tools=list(tools.values()))
            request = await hook_runner.before_model_request(request, None)
            capabilities = await agent.model.capabilities()
            prepared = await self.context_manager.prepare(request, capabilities)
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
            async for step_event in self._iter_provider_stream_events(agent.model.stream(prepared.request)):
                if step_event.kind == "text_delta" and step_event.delta:
                    event = RuntimeEvent(
                        type="model.stream.delta",
                        run_id=run_id,
                        agent_name=agent.name,
                        payload={"delta": step_event.delta},
                    )
                    await self.event_sink.emit(event)
                    yield event
                    continue

                if step_event.kind == "tool_call_completed" and step_event.call_id and step_event.name:
                    call = ToolCall(id=step_event.call_id, name=step_event.name, arguments=step_event.arguments)
                    call = await hook_runner.before_tool_call(call, None)
                    await self.event_sink.emit(
                        RuntimeEvent(
                            type="tool.started",
                            run_id=run_id,
                            agent_name=agent.name,
                            payload={"name": call.name},
                        )
                    )
                    result = await execute_tool_call(call, tools, ToolContext(run_id=run_id))
                    result = await hook_runner.after_tool_result(result, None)
                    await self.event_sink.emit(
                        RuntimeEvent(
                            type="tool.completed",
                            run_id=run_id,
                            agent_name=agent.name,
                            payload={"name": result.name},
                        )
                    )
                    messages.append(
                        Message.tool(
                            result.content,
                            metadata={"tool_call_id": result.call_id, "name": result.name},
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
                continue

            if saw_message_completed:
                await self.event_sink.emit(
                    RuntimeEvent(
                        type="run.completed",
                        run_id=run_id,
                        agent_name=agent.name,
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
