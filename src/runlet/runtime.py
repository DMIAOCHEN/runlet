from __future__ import annotations

from uuid import uuid4

from runlet.context import ContextManager
from runlet.core import Agent, Message, RunResult
from runlet.events import EventSink, InMemoryObserver, RuntimeEvent
from runlet.hooks import HookRunner
from runlet.models import ModelRequest
from runlet.policies import RunPolicy
from runlet.tools import ToolContext, ToolSpec, execute_tool_call


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
        await self.event_sink.emit(
            RuntimeEvent(
                type="run.started",
                run_id=run_id,
                agent_name=agent.name,
                payload={"input": input},
            )
        )

        request = ModelRequest(messages=[Message.system(agent.instructions), Message.user(input)])
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
        async for chunk in agent.model.stream(prepared.request):
            if chunk.delta:
                event = RuntimeEvent(
                    type="model.stream.delta",
                    run_id=run_id,
                    agent_name=agent.name,
                    payload={"delta": chunk.delta},
                )
                await self.event_sink.emit(event)
                yield event
        await self.event_sink.emit(
            RuntimeEvent(
                type="model.stream.completed",
                run_id=run_id,
                agent_name=agent.name,
            )
        )
        await self.event_sink.emit(
            RuntimeEvent(
                type="run.completed",
                run_id=run_id,
                agent_name=agent.name,
            )
        )
