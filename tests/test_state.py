import unittest

from runlet.state import InMemoryStateStore, StateScope


class StateTests(unittest.IsolatedAsyncioTestCase):
    async def test_in_memory_state_store_round_trips_conversation_state(self) -> None:
        store = InMemoryStateStore()
        scope = StateScope(kind="conversation", key="conv_1")

        await store.save(scope, {"count": 1})
        loaded = await store.load(scope)

        self.assertEqual(loaded["count"], 1)

    async def test_loaded_state_is_copy(self) -> None:
        store = InMemoryStateStore()
        scope = StateScope(kind="run", key="run_1")

        await store.save(scope, {"count": 1})
        loaded = await store.load(scope)
        loaded["count"] = 2

        self.assertEqual((await store.load(scope))["count"], 1)
