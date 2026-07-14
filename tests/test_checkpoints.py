import unittest

import runlet
import runlet.runtime as runtime
from runlet.core import HumanRequest, Message, ModelRequest, ToolCall, Usage
from runlet.runtime.checkpoints import InMemoryCheckpointStore, RunCheckpoint


def make_checkpoint(checkpoint_id: str) -> RunCheckpoint:
    tool_call = ToolCall(id="call_1", name="search")
    request = ModelRequest(messages=[Message.user("Find the answer.")])
    return RunCheckpoint(
        id=checkpoint_id,
        run_id="run_1",
        agent_name="assistant",
        request=request,
        messages=(Message.user("Find the answer."),),
        pending_request=HumanRequest(
            id="request_1",
            kind="tool_approval",
            prompt="Approve the tool call?",
            tool_call=tool_call,
        ),
        pending_tool_call=tool_call,
        step=1,
        reasoning="I need to search.",
        usage=Usage(),
    )


class InMemoryCheckpointStoreTests(unittest.IsolatedAsyncioTestCase):
    def test_checkpoint_types_are_exported_from_runtime_and_root_apis(self) -> None:
        expected = {"RunCheckpoint", "CheckpointStore", "InMemoryCheckpointStore"}

        self.assertTrue(expected.issubset(set(runtime.__all__)))
        self.assertTrue(expected.issubset(set(runlet.__all__)))

    async def test_in_memory_checkpoint_store_round_trips_and_deletes(self) -> None:
        store = InMemoryCheckpointStore()
        checkpoint = make_checkpoint("checkpoint_1")

        await store.save(checkpoint)

        self.assertEqual(await store.load("checkpoint_1"), checkpoint)

        await store.delete("checkpoint_1")

        self.assertIsNone(await store.load("checkpoint_1"))
