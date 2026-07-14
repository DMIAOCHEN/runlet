from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
import json
from uuid import uuid4

from runlet.core.agent import Agent
from runlet.core.events import EventSink, InMemoryObserver, RuntimeEvent
from runlet.core.human import HumanRequest, HumanResponse
from runlet.core.messages import Message, ToolCall
from runlet.core.models import ModelRequest, ModelStreamEvent, ProviderStreamEvent
from runlet.core.runs import RunResult, Usage
from runlet.integrations.hooks import HookRunner
from runlet.integrations.human import HumanInputToolSpec
from runlet.integrations.tools import ToolContext, ToolSpec, execute_tool_call
from runlet.integrations.tools import validate_arguments
from runlet.runtime.checkpoints import CheckpointStore, InMemoryCheckpointStore, RunCheckpoint
from runlet.runtime.context import ContextManager
from runlet.runtime.policies import RunPolicy


class Runtime:
    def __init__(
        self,
        event_sink: EventSink | None = None,
        context_manager: ContextManager | None = None,
        policy: RunPolicy | None = None,
        checkpoint_store: CheckpointStore | None = None,
    ) -> None:
        self.event_sink = event_sink or InMemoryObserver()
        self.context_manager = context_manager or ContextManager()
        self.policy = policy or RunPolicy()
        self.checkpoint_store = checkpoint_store or InMemoryCheckpointStore()

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
        return await self._run_loop(
            agent,
            request,
            run_id,
            messages,
            tools,
            0,
            [],
            Usage(),
            hook_runner,
        )

    async def resume(
        self,
        agent: Agent,
        *,
        checkpoint_id: str,
        response: HumanResponse,
    ) -> RunResult:
        checkpoint = await self.checkpoint_store.load(checkpoint_id)
        if checkpoint is None:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        if agent.name != checkpoint.agent_name:
            raise ValueError("Checkpoint agent does not match the resumed agent.")

        request = checkpoint.pending_request
        if request is None:
            tools = self._agent_tools(agent)
            return await self._continue_checkpoint(
                agent,
                checkpoint,
                list(checkpoint.messages),
                tools,
                HookRunner(agent.hooks),
            )

        messages = list(checkpoint.messages)
        tools = self._agent_tools(agent)
        call = checkpoint.pending_tool_call
        if call is None:
            hook_runner = HookRunner(agent.hooks)
            remaining_result = await self._process_remaining_calls(
                agent,
                checkpoint.request,
                checkpoint.run_id,
                messages,
                tools,
                checkpoint.step,
                [checkpoint.reasoning] if checkpoint.reasoning else [],
                checkpoint.usage,
                hook_runner,
                checkpoint.pending_tool_calls,
                checkpoint,
            )
            if remaining_result is not None:
                return remaining_result
            return await self._continue_checkpoint(agent, checkpoint, messages, tools, hook_runner)
        try:
            self._validate_human_response(request, response)
            if request.kind == "tool_approval" and response.action == "approve":
                arguments = response.edited_arguments if response.edited_arguments is not None else call.arguments
                tool = tools.get(call.name)
                if tool is None:
                    raise ValueError(f"Tool not found: {call.name}")
                if not isinstance(arguments, Mapping):
                    raise ValueError("Approved tool arguments must be a mapping.")
                try:
                    approved_arguments = dict(arguments)
                except (TypeError, ValueError) as error:
                    raise ValueError("Approved tool arguments must be convertible to a mapping.") from error
                validate_arguments(tool.input_schema, approved_arguments)
                call = replace(call, arguments=approved_arguments)
        except ValueError:
            await self.event_sink.emit(
                RuntimeEvent(
                    type="human.response_rejected",
                    run_id=checkpoint.run_id,
                    agent_name=agent.name,
                    payload=self._human_event_payload(request, checkpoint, action=response.action),
                )
            )
            raise

        if request.kind == "tool_approval":
            if response.action == "reject":
                messages.append(
                    Message.tool(
                        "Human rejected tool call.",
                        metadata={
                            "tool_call_id": call.id,
                            "name": call.name,
                            "human_rejected": True,
                            "reason": response.metadata.get("reason"),
                        },
                    )
                )
        else:
            messages.append(
                Message.tool(
                    json.dumps({"kind": request.kind, "value": response.value}, separators=(",", ":")),
                    metadata={"tool_call_id": call.id, "name": call.name, "human_input": True},
                )
            )

        await self.event_sink.emit(
            RuntimeEvent(
                type="human.responded",
                run_id=checkpoint.run_id,
                agent_name=agent.name,
                payload=self._human_event_payload(request, checkpoint, action=response.action),
            )
        )
        await self.event_sink.emit(
            RuntimeEvent(
                type="run.resumed",
                run_id=checkpoint.run_id,
                agent_name=agent.name,
                payload={"checkpoint_id": checkpoint.id},
            )
        )
        hook_runner = HookRunner(agent.hooks)
        if request.kind == "tool_approval" and response.action == "approve":
            await self._execute_call(agent, checkpoint.run_id, call, tools, messages, hook_runner)
        await self._save_continuation(checkpoint, messages, checkpoint.pending_tool_calls)

        remaining_result = await self._process_remaining_calls(
            agent,
            checkpoint.request,
            checkpoint.run_id,
            messages,
            tools,
            checkpoint.step,
            [checkpoint.reasoning] if checkpoint.reasoning else [],
            checkpoint.usage,
            hook_runner,
            checkpoint.pending_tool_calls,
            checkpoint,
        )
        if remaining_result is not None:
            return remaining_result

        return await self._continue_checkpoint(
            agent,
            checkpoint,
            messages,
            tools,
            hook_runner,
        )

    async def _run_loop(
        self,
        agent: Agent,
        request: ModelRequest,
        run_id: str,
        messages: list[Message],
        tools: dict[str, ToolSpec],
        start_step: int,
        reasoning: list[str],
        usage: Usage,
        hook_runner: HookRunner,
    ) -> RunResult:
        for step in range(start_step, self.policy.max_steps):
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
                reasoning.append(response.reasoning)
            usage = self._add_usage(usage, response.usage)
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
                for call_index, call in enumerate(response.tool_calls):
                    call = await hook_runner.before_tool_call(call, None)
                    tool = tools.get(call.name)
                    if isinstance(tool, HumanInputToolSpec):
                        human_request = replace(tool.human_request_from_call(call), tool_call=call)
                        return await self._interrupt(
                            agent,
                            request,
                            run_id,
                            messages,
                            human_request,
                            call,
                            step + 1,
                            reasoning,
                            usage,
                            tuple(response.tool_calls[call_index + 1 :]),
                        )
                    if tool is not None and tool.requires_approval:
                        human_request = HumanRequest(
                            id=f"hta_{uuid4().hex}",
                            kind="tool_approval",
                            prompt=f"Approve tool call: {call.name}",
                            tool_call=call,
                        )
                        return await self._interrupt(
                            agent,
                            request,
                            run_id,
                            messages,
                            human_request,
                            call,
                            step + 1,
                            reasoning,
                            usage,
                            tuple(response.tool_calls[call_index + 1 :]),
                        )
                    await self._execute_call(agent, run_id, call, tools, messages, hook_runner)
                continue

            run_result = RunResult.completed(
                run_id=run_id,
                output=response.message.text,
                reasoning="".join(reasoning),
                usage=usage,
                messages=tuple(messages),
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
            reasoning="".join(reasoning),
            usage=usage,
            messages=tuple(messages),
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

    async def _interrupt(
        self,
        agent: Agent,
        request: ModelRequest,
        run_id: str,
        messages: list[Message],
        human_request: HumanRequest,
        call: ToolCall,
        next_step: int,
        reasoning: list[str],
        usage: Usage,
        pending_tool_calls: tuple[ToolCall, ...] = (),
        replace_checkpoint_id: str | None = None,
        yielded_events: list[RuntimeEvent] | None = None,
    ) -> RunResult:
        checkpoint_id = f"checkpoint_{uuid4().hex}"
        checkpoint = RunCheckpoint(
            id=checkpoint_id,
            run_id=run_id,
            agent_name=agent.name,
            request=self._checkpoint_request(request),
            messages=tuple(messages),
            pending_request=human_request,
            pending_tool_call=call,
            step=next_step,
            reasoning="".join(reasoning),
            usage=usage,
            pending_tool_calls=pending_tool_calls,
        )
        await self.checkpoint_store.save(checkpoint)
        if replace_checkpoint_id is not None:
            await self.checkpoint_store.delete(replace_checkpoint_id)
        interruption_events = (
            RuntimeEvent(
                type="human.requested",
                run_id=run_id,
                agent_name=agent.name,
                payload=self._human_event_payload(human_request, checkpoint),
            ),
            RuntimeEvent(
                type="run.interrupted",
                run_id=run_id,
                agent_name=agent.name,
                payload={"checkpoint_id": checkpoint_id, "kind": human_request.kind},
            ),
        )
        for event in interruption_events:
            await self.event_sink.emit(event)
        if yielded_events is not None:
            yielded_events.extend(interruption_events)
        return RunResult.interrupted(
            run_id=run_id,
            interruption=human_request,
            checkpoint_id=checkpoint_id,
            reasoning="".join(reasoning),
            usage=usage,
            messages=tuple(messages),
        )

    async def _process_remaining_calls(
        self,
        agent: Agent,
        request: ModelRequest,
        run_id: str,
        messages: list[Message],
        tools: dict[str, ToolSpec],
        step: int,
        reasoning: list[str],
        usage: Usage,
        hook_runner: HookRunner,
        pending_calls: tuple[ToolCall, ...],
        checkpoint: RunCheckpoint,
    ) -> RunResult | None:
        for call_index, call in enumerate(pending_calls):
            call = await hook_runner.before_tool_call(call, None)
            tool = tools.get(call.name)
            if isinstance(tool, HumanInputToolSpec):
                human_request = replace(tool.human_request_from_call(call), tool_call=call)
                return await self._interrupt(
                    agent,
                    request,
                    run_id,
                    messages,
                    human_request,
                    call,
                    step,
                    reasoning,
                    usage,
                    pending_calls[call_index + 1 :],
                    checkpoint.id,
                )
            if tool is not None and tool.requires_approval:
                human_request = HumanRequest(
                    id=f"hta_{uuid4().hex}",
                    kind="tool_approval",
                    prompt=f"Approve tool call: {call.name}",
                    tool_call=call,
                )
                return await self._interrupt(
                    agent,
                    request,
                    run_id,
                    messages,
                    human_request,
                    call,
                    step,
                    reasoning,
                    usage,
                    pending_calls[call_index + 1 :],
                    checkpoint.id,
                )
            await self._execute_call(agent, run_id, call, tools, messages, hook_runner)
            await self._save_continuation(checkpoint, messages, pending_calls[call_index + 1 :])
        return None

    async def _execute_call(
        self,
        agent: Agent,
        run_id: str,
        call: ToolCall,
        tools: dict[str, ToolSpec],
        messages: list[Message],
        hook_runner: HookRunner,
    ) -> None:
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

    async def _save_continuation(
        self,
        checkpoint: RunCheckpoint,
        messages: list[Message],
        pending_tool_calls: tuple[ToolCall, ...],
    ) -> None:
        await self.checkpoint_store.save(
            replace(
                checkpoint,
                messages=tuple(messages),
                pending_tool_call=None,
                pending_tool_calls=pending_tool_calls,
            )
        )

    async def _continue_checkpoint(
        self,
        agent: Agent,
        checkpoint: RunCheckpoint,
        messages: list[Message],
        tools: dict[str, ToolSpec],
        hook_runner: HookRunner,
    ) -> RunResult:
        ready_checkpoint = replace(
            checkpoint,
            messages=tuple(messages),
            pending_request=None,
            pending_tool_call=None,
            pending_tool_calls=(),
        )
        await self.checkpoint_store.save(ready_checkpoint)
        result = await self._run_loop(
            agent,
            checkpoint.request,
            checkpoint.run_id,
            messages,
            tools,
            checkpoint.step,
            [checkpoint.reasoning] if checkpoint.reasoning else [],
            checkpoint.usage,
            hook_runner,
        )
        await self.checkpoint_store.delete(checkpoint.id)
        return result

    @staticmethod
    def _add_usage(total: Usage, response: Usage) -> Usage:
        return Usage(
            input_tokens=total.input_tokens + response.input_tokens,
            output_tokens=total.output_tokens + response.output_tokens,
            source=response.source,
        )

    @staticmethod
    def _human_event_payload(
        request: HumanRequest,
        checkpoint: RunCheckpoint,
        *,
        action: object | None = None,
    ) -> dict[str, str]:
        payload = {
            "request_id": request.id,
            "checkpoint_id": checkpoint.id,
            "kind": request.kind,
        }
        if isinstance(action, str) and action in {"approve", "reject", "select", "submit"}:
            payload["action"] = action
        return payload

    def _validate_human_response(self, request: HumanRequest, response: HumanResponse) -> None:
        if response.request_id != request.id:
            raise ValueError("Human response request id does not match the pending request.")
        if not isinstance(response.action, str):
            raise ValueError("Human response action must be a string.")
        if response.edited_arguments is not None and (request.kind != "tool_approval" or response.action != "approve"):
            raise ValueError("edited_arguments are only allowed when approving a tool call.")
        if request.kind == "tool_approval":
            if response.action not in {"approve", "reject"}:
                raise ValueError("Tool approval responses must approve or reject.")
            return
        if request.kind == "choice":
            if response.action != "select":
                raise ValueError("Choice responses must select an option.")
            if not isinstance(response.value, str) or response.value not in {option.id for option in request.options}:
                raise ValueError("Choice response must select a listed option.")
            return
        if response.action != "submit" or not isinstance(response.value, str):
            raise ValueError("Input responses must submit a string value.")

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

        for step in range(self.policy.max_steps):
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
            collected_deltas: list[str] = []
            collected_reasoning_deltas: list[str] = []
            completed_tool_calls: list[dict[str, object]] = []
            completed_calls: list[ToolCall] = []
            pending_tool_messages: list[Message] = []
            pending_human_request: HumanRequest | None = None
            pending_human_call: ToolCall | None = None
            pending_human_call_index: int | None = None
            provider_stream = agent.model.stream(prepared.request)
            stream_events = self._iter_provider_stream_events(provider_stream)
            async for step_event in stream_events:
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
                    completed_calls.append(call)
                    completed_tool_calls.append(
                        {"id": call.id, "name": call.name, "arguments": dict(call.arguments)}
                    )
                    if pending_human_request is not None:
                        continue

                    tool = tools.get(call.name)
                    if isinstance(tool, HumanInputToolSpec):
                        pending_human_request = replace(tool.human_request_from_call(call), tool_call=call)
                        pending_human_call = call
                        pending_human_call_index = len(completed_calls) - 1
                        continue
                    if tool is not None and tool.requires_approval:
                        pending_human_request = HumanRequest(
                            id=f"hta_{uuid4().hex}",
                            kind="tool_approval",
                            prompt=f"Approve tool call: {call.name}",
                            tool_call=call,
                        )
                        pending_human_call = call
                        pending_human_call_index = len(completed_calls) - 1
                        continue

                    await self._execute_call(agent, run_id, call, tools, pending_tool_messages, hook_runner)
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

            if completed_calls:
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
                if (
                    pending_human_request is not None
                    and pending_human_call is not None
                    and pending_human_call_index is not None
                ):
                    interruption_events: list[RuntimeEvent] = []
                    await self._interrupt(
                        agent,
                        request,
                        run_id,
                        messages,
                        pending_human_request,
                        pending_human_call,
                        step + 1,
                        all_reasoning_deltas,
                        Usage(),
                        tuple(completed_calls[pending_human_call_index + 1 :]),
                        yielded_events=interruption_events,
                    )
                    for event in interruption_events:
                        yield event
                    return
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

    @staticmethod
    def _agent_tools(agent: Agent) -> dict[str, ToolSpec]:
        return {tool_spec.name: tool_spec for tool_spec in agent.tools if isinstance(tool_spec, ToolSpec)}

    @staticmethod
    def _checkpoint_request(request: ModelRequest) -> ModelRequest:
        """Persist request data without executable tool handlers."""
        return ModelRequest(
            messages=[Message(message.role, message.text, dict(message.metadata)) for message in request.messages],
            tools=[],
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
