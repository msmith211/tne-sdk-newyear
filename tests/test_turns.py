"""Turn handling: deadlines, fallbacks, automatic waits, action marks, the
grounding refresh, the memory mirror and reviews after the action."""
import asyncio
import json
import sys
import threading
from pathlib import Path

import httpx
import pytest
from websockets.asyncio.server import serve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tne_sdk.agent as agent_module
import tne_sdk.mcp_server as mcp
from tne_sdk import turns
from tne_sdk.agent import Agent
from tne_sdk.catalog import fetch_catalog, grounding_block, is_grounding_tick
from tne_sdk.client import TNEClient
from tne_sdk.config import AgentConfig
from tne_sdk.memory.null_memory import NullMemory


def _state(tick=7, **fields):
    return {"tick_info": {"current_tick": tick}, "integrity": 80, "max_integrity": 100, **fields}


# ── Pure helpers ────────────────────────────────────────────────────────── #

class TestBudget:
    def test_it_is_what_is_left_before_the_deadline(self):
        state = _state(deadline_at_ms=160_000, server_now_ms=100_000)
        assert turns.DecisionBudget.from_state(state).seconds == 57.0
        assert turns.DecisionBudget.from_state(state, waited=20, margin=5).seconds == 35.0

    def test_it_never_goes_below_zero(self):
        budget = turns.DecisionBudget.from_state(_state(deadline_at_ms=1000, server_now_ms=1000), waited=9)
        assert budget.seconds == 0 and budget.spent

    def test_an_older_server_gives_no_deadline(self):
        budget = turns.DecisionBudget.from_state(_state())
        assert budget.seconds is None and not budget.spent


class TestFallback:
    def test_out_of_combat_it_waits(self):
        action = turns.fallback_action(_state())
        assert action["action"] == "wait" and action["decision_source"] == "fallback_rule"
        assert action["reasoning"].startswith("[fallback_rule]")

    def test_in_combat_it_defends(self):
        assert turns.fallback_action(_state(combat_state={"in_combat": True}))["action"] == "defend"

    def test_in_combat_and_badly_hurt_it_flees(self):
        state = _state(combat_state={"in_combat": True}, integrity=20)
        assert turns.fallback_action(state)["action"] == "flee"
        assert turns.fallback_action({**state, "integrity": 25})["action"] == "defend"


_TRAVEL = {"from": "rust_wastes", "to": "archive_node", "destination": "deep_loop", "ticks_left": 2, "paused": None}


class TestAutoWait:
    def test_travelling_with_nothing_new_needs_no_model(self):
        reason = turns.auto_wait_reason(_state(travel=_TRAVEL))
        assert reason == "Travelling rust_wastes → archive_node, 2 ticks left, bound for deep_loop."
        action = turns.auto_wait(reason)
        assert action["action"] == "wait" and action["decision_source"] == "auto_wait"

    @pytest.mark.parametrize("changes", [
        {"travel": None},
        {"travel": {**_TRAVEL, "paused": "encounter"}},
        {"combat_state": {"in_combat": True}},
        {"messages": [{"content": "hello"}]},
        {"last_action_result": {"status": "failed"}},
        {"last_action_result": {"status": "stale"}},
    ])
    def test_anything_new_goes_to_the_model(self, changes):
        assert turns.auto_wait_reason(_state(**{"travel": _TRAVEL, **changes})) is None

    def test_a_good_last_result_is_nothing_new(self):
        assert turns.auto_wait_reason(_state(travel=_TRAVEL, last_action_result={"status": "success"}))


class TestStamp:
    def test_an_action_names_its_tick_and_gets_an_id(self):
        action = {"action": "wait", "reasoning": "r", "decision_source": "model"}
        stamped = turns.stamp(action, _state(42), "abcd1234", 3)
        assert stamped == {"action": "wait", "reasoning": "r", "based_on_tick": 42, "action_id": "abcd1234-42-3"}
        assert "decision_source" in action          # the original is left alone

    def test_a_lean_state_has_its_tick_too(self):
        assert turns.stamp({"action": "wait"}, {"tick": 9}, "s", 1)["based_on_tick"] == 9

    def test_no_tick_no_claim(self):
        stamped = turns.stamp({"action": "wait"}, {}, "s", 1)
        assert "based_on_tick" not in stamped and stamped["action_id"] == "s-x-1"

    def test_session_ids_differ(self):
        assert turns.new_session_id() != turns.new_session_id()


class TestMemoryPatch:
    def test_tasks_become_the_plan_and_goals(self):
        tasks = [{"description": f"Task {n}"} for n in range(7)]
        patch = turns.server_memory_patch(tasks, "Selling scrap to fund a blade.")
        assert patch["goals"] == [f"Task {n}" for n in range(5)]
        assert patch["plan"] == "Task 0 | Selling scrap to fund a blade."

    def test_nothing_to_say(self):
        assert turns.server_memory_patch([]) == {"plan": None, "goals": []}

    def test_text_is_kept_inside_the_server_limits(self):
        patch = turns.server_memory_patch([{"description": "g " * 200}], "n" * 900)
        assert len(patch["goals"][0]) <= 120 and len(patch["plan"]) <= 400

    def test_repeated_tasks_are_listed_once(self):
        assert turns.server_memory_patch([{"description": "Same"}, {"description": "Same"}])["goals"] == ["Same"]


# ── The grounding refresh ───────────────────────────────────────────────── #

_CATALOG = {
    "catalog_version": "1.0.35+abc",
    "game": {"briefing": "You are an autonomous AI agent in NULL EPOCH."},
    "factions": {"free_processes": {"name": "Free Processes", "description": "No masters.",
                                    "bonus_description": "Travel costs -25%.", "home_territory": "free_haven"}},
    "territories": {"rust_wastes": {"name": "Rust Wastes", "danger": 3, "lore": "Industrial ruins."}},
    "actions": {"gather": {"does": "Harvest a node.", "parameters": "`node_id` (str).",
                           "notes": "Yield falls with repeats.", "timing": "runs up to 1 tick late"},
                "wait": {"does": "Do nothing this tick."},
                "craft": {"does": "Make an item."}},
}


class TestGrounding:
    def test_it_comes_round_every_nth_tick(self):
        assert [t for t in range(0, 31) if is_grounding_tick(t, 10)] == [10, 20, 30]
        assert not any(is_grounding_tick(t, 0) for t in range(31))
        assert not is_grounding_tick("?", 10)

    def test_the_block_says_what_the_game_is_and_how_the_actions_work(self):
        state = _state(faction="free_processes", current_territory="rust_wastes",
                       available_actions=[{"action": "gather"}, {"action": "wait"}, {"action": "dance"}])
        block = grounding_block(_CATALOG, state)
        assert block.startswith("=== GROUNDING REFRESH")
        assert "You are an autonomous AI agent in NULL EPOCH." in block
        assert "YOUR FACTION: Free Processes No masters. Faction bonus: Travel costs -25%." in block
        assert "WHERE YOU ARE: Rust Wastes (danger 3). Industrial ruins." in block
        assert "  gather: Harvest a node. Parameters: `node_id` (str). Yield falls with repeats." in block
        assert "  wait: Do nothing this tick." in block
        assert "craft" not in block and "dance" not in block       # only what is on offer now
        assert block.rstrip().endswith("Your current situation follows. ===")

    def test_no_catalogue_no_block(self):
        assert grounding_block(None, _state()) == "" and grounding_block({}, _state()) == ""

    def test_the_catalogue_is_fetched_with_the_key(self, monkeypatch):
        seen = {}

        def fake_get(url, headers=None, timeout=None):
            seen.update(url=url, headers=headers)
            return httpx.Response(200, json=_CATALOG, headers={"etag": '"v1"'}, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx, "get", fake_get)
        catalog, etag = fetch_catalog("game.test", "key-1", secure=True, etag='"v0"')
        assert catalog == _CATALOG and etag == '"v1"'
        assert seen["url"] == "https://game.test/v1/agent/catalog"
        assert seen["headers"] == {"Authorization": "Bearer key-1", "If-None-Match": '"v0"'}

    @pytest.mark.parametrize("response", [
        lambda url: httpx.Response(404, json={"detail": "Not Found"}, request=httpx.Request("GET", url)),
        lambda url: httpx.Response(200, json=["not", "a", "catalogue"], request=httpx.Request("GET", url)),
    ])
    def test_a_server_without_one_is_fine(self, monkeypatch, response):
        monkeypatch.setattr(httpx, "get", lambda url, **k: response(url))
        assert fetch_catalog("old.test", "k") == (None, None)

    def test_a_network_error_is_fine(self, monkeypatch):
        def boom(*a, **k):
            raise httpx.ConnectError("down")
        monkeypatch.setattr(httpx, "get", boom)
        assert fetch_catalog("down.test", "k") == (None, None)

    def test_unchanged_is_said(self, monkeypatch):
        monkeypatch.setattr(httpx, "get", lambda url, **k: httpx.Response(304, request=httpx.Request("GET", url)))
        assert fetch_catalog("game.test", "k", etag='"v1"') == (None, '"v1"')


# ── The agent ───────────────────────────────────────────────────────────── #

class _Client:
    """Stands in for TNEClient: records what is sent mid-turn."""
    host, api_key, secure = "game.test", "k", False

    def __init__(self):
        self.sent = []

    async def send(self, action):
        self.sent.append(action)
        return True

    def seconds_since_received(self, state):
        return 0.0

    async def run(self, on_tick, on_message=None, after_turn=None):
        raise AssertionError("not used")


class _Model:
    """An LLM that answers with a fixed action, optionally after a delay."""

    def __init__(self, answer=None, delay=0.0):
        self.answer = answer or {"action": "gather", "parameters": {"node_id": "node_scrap_1"}, "reasoning": "r"}
        self.delay = delay
        self.prompts = []
        self.calls = []

    def chat_completion(self, messages, **kwargs):
        self.prompts.append(messages[-1].content)
        self.calls.append(messages[0].content[:40])
        if self.delay:
            threading.Event().wait(self.delay)
        return json.dumps(self.answer)


class _Memory(NullMemory):
    def __init__(self, tasks=(), narrative=""):
        super().__init__()
        self._tasks = [{"task_id": n + 1, "parent_id": None, "status": "pending", "priority": 10, **t}
                       for n, t in enumerate(tasks)]
        self._narrative = narrative

    def get_active_tasks(self, limit=15):
        return self._tasks[:limit]

    def get_knowledge(self, key):
        if key == "summary:narrative":
            return self._narrative
        return super().get_knowledge(key)


_GATHER = {"action": "gather", "parameters": {"node_id": {"valid_values": ["node_scrap_1"]}}}
_WAIT = {"action": "wait", "parameters": {}}


def _agent(model=None, client=None, memory=None, **config):
    cfg = AgentConfig(reflection_cooldown_ticks=10_000, tactical_review_cooldown_ticks=10_000, **config)
    return Agent(cfg, client or _Client(), memory or _Memory(), model or _Model())


def _turn(agent, state):
    return asyncio.run(agent._on_tick(state))


def _playable(tick=7, **fields):
    return _state(tick, available_actions=[_GATHER, _WAIT, {"action": "flee", "parameters": {}}],
                  current_territory="rust_wastes", faction="free_processes",
                  nearby_nodes=[{"node_id": "node_scrap_1", "resource": "scrap_metal", "track": "scavenging",
                                 "required_level": 1, "your_skill": 1, "can_gather": True,
                                 "ticks_until_ready": 0}],
                  **fields)


class TestAgentTurns:
    def test_every_action_names_its_tick_and_has_an_id(self):
        agent = _agent()
        first = _turn(agent, _playable(7))
        second = _turn(agent, _playable(8))
        assert first["action"] == "gather" and first["based_on_tick"] == 7 and second["based_on_tick"] == 8
        assert first["action_id"] != second["action_id"]
        assert first["action_id"].startswith(agent._session_id) and "decision_source" not in first

    def test_travelling_with_nothing_new_calls_no_model(self):
        model = _Model()
        agent = _agent(model)
        action = _turn(agent, _playable(7, travel=_TRAVEL))
        assert action["action"] == "wait" and action["reasoning"].startswith("[auto_wait] Travelling")
        assert action["based_on_tick"] == 7 and model.prompts == [] and agent.auto_waits == 1

    def test_an_interception_goes_to_the_model(self):
        model = _Model()
        agent = _agent(model)
        _turn(agent, _playable(7, travel={**_TRAVEL, "paused": "encounter"}))
        assert len(model.prompts) == 1 and agent.auto_waits == 0

    def test_automatic_waits_can_be_turned_off(self):
        model = _Model()
        _turn(_agent(model, auto_wait_travel=False), _playable(7, travel=_TRAVEL))
        assert len(model.prompts) == 1

    def test_a_model_in_time_sends_nothing_extra(self):
        client = _Client()
        agent = _agent(client=client)
        action = _turn(agent, _playable(7, deadline_at_ms=160_000, server_now_ms=100_000))
        assert action["action"] == "gather" and client.sent == [] and agent.fallbacks_sent == 0

    def test_a_slow_model_gets_a_fallback_at_the_deadline_and_still_answers(self):
        client = _Client()
        agent = _agent(_Model({"action": "flee", "reasoning": "too strong"}, delay=0.4), client,
                       deadline_margin_seconds=0.0)
        state = _playable(7, deadline_at_ms=100_100, server_now_ms=100_000,     # 0.1 s to decide
                          combat_state={"in_combat": True})
        action = _turn(agent, state)
        (fallback,) = client.sent
        assert fallback["action"] == "defend" and fallback["reasoning"].startswith("[fallback_rule]")
        assert fallback["based_on_tick"] == 7 and "decision_source" not in fallback
        assert action["action"] == "flee" and action["based_on_tick"] == 7       # the model's own, sent late
        assert action["action_id"] != fallback["action_id"] and agent.fallbacks_sent == 1

    def test_no_deadline_from_the_server_means_no_fallback(self):
        client = _Client()
        action = _turn(_agent(_Model(delay=0.1), client), _playable(7))
        assert action["action"] == "gather" and client.sent == []

    def test_fallbacks_can_be_turned_off(self):
        client = _Client()
        agent = _agent(_Model(delay=0.2), client, deadline_fallback=False, deadline_margin_seconds=0.0)
        _turn(agent, _playable(7, deadline_at_ms=100_050, server_now_ms=100_000))
        assert client.sent == []

    def test_the_prompt_is_refreshed_every_tenth_tick(self):
        model = _Model()
        agent = _agent(model)
        agent._catalog = _CATALOG
        for tick in (9, 10, 11):
            _turn(agent, _playable(tick))
        assert [p.startswith("=== GROUNDING REFRESH") for p in model.prompts] == [False, True, False]
        refreshed = model.prompts[1]
        assert "RULES FOR THE ACTIONS AVAILABLE NOW:" in refreshed and "=== TICK 10 ===" in refreshed
        assert refreshed.index("=== END OF REFRESH") < refreshed.index("=== TICK 10 ===")
        assert model.prompts[0] in ("",) or "=== TICK 9 ===" in model.prompts[0]

    def test_without_a_catalogue_the_prompt_is_as_it_was(self):
        model = _Model()
        _turn(_agent(model), _playable(10))
        assert not model.prompts[0].startswith("=== GROUNDING REFRESH")

    def test_the_refresh_can_be_turned_off(self):
        model = _Model()
        agent = _agent(model, grounding_every_ticks=0)
        agent._catalog = _CATALOG
        _turn(agent, _playable(10))
        assert not model.prompts[0].startswith("=== GROUNDING REFRESH")

    def test_a_stale_result_reaches_the_model(self):
        model = _Model()
        agent = _agent(model)
        _turn(agent, _playable(8, last_action_result={
            "action": "buy", "status": "stale", "tick": 7,
            "summary": "Action 'buy' was not run: it was decided on the state of tick 5."}))
        assert "STALE" in model.prompts[0].upper() and "buy" in model.prompts[0]
        assert "STALE ACTIONS" in agent.system_prompt


class TestMemoryMirror:
    _SERVER = {"plan": None, "goals": [], "notes": []}

    def _agent(self, **config):
        memory = _Memory([{"description": "Buy a blade"}, {"description": "Reach level 3"}], "Short of credits.")
        return _agent(memory=memory, **config)

    def test_the_plan_and_goals_ride_on_the_action(self):
        action = _turn(self._agent(), _playable(7, memory=self._SERVER))
        assert action["memory"] == {"plan": "Buy a blade | Short of credits.",
                                    "goals": ["Buy a blade", "Reach level 3"]}
        assert "notes" not in action["memory"]          # the server's notes are left alone

    def test_it_is_not_sent_again_until_it_changes(self):
        agent = self._agent()
        assert "memory" in _turn(agent, _playable(7, memory=self._SERVER))
        assert "memory" not in _turn(agent, _playable(8, memory=self._SERVER))
        agent.memory._tasks.insert(0, {"task_id": 9, "parent_id": None, "status": "pending", "priority": 99,
                                       "description": "Flee the wraith"})
        assert _turn(agent, _playable(9, memory=self._SERVER))["memory"]["goals"][0] == "Flee the wraith"

    def test_nothing_is_sent_when_the_server_already_has_it(self):
        held = {"plan": "Buy a blade | Short of credits.", "goals": ["Buy a blade", "Reach level 3"], "notes": ["x"]}
        assert "memory" not in _turn(self._agent(), _playable(7, memory=held))

    def test_a_server_without_memory_is_sent_none(self):
        assert "memory" not in _turn(self._agent(), _playable(7))

    def test_it_can_be_turned_off(self):
        assert "memory" not in _turn(self._agent(mirror_memory=False), _playable(7, memory=self._SERVER))

    def test_an_agent_with_no_local_memory_sends_nothing(self):
        assert "memory" not in _turn(_agent(memory=NullMemory()), _playable(7, memory=self._SERVER))


class TestReviewsAfterTheAction:
    def _reviewing_agent(self, monkeypatch, **config):
        ran = []

        async def tactical(self, state):
            ran.append(("tactical", turns.tick_of(state)))

        async def reflection(self, tick):
            ran.append(("reflection", tick))

        monkeypatch.setattr(Agent, "_run_tactical_review", tactical)
        monkeypatch.setattr(Agent, "_run_reflection", reflection)
        cfg = AgentConfig(**config)
        return Agent(cfg, _Client(), _Memory(), _Model()), ran

    def test_a_review_no_longer_takes_the_turn(self, monkeypatch):
        agent, ran = self._reviewing_agent(monkeypatch, tactical_review_cooldown_ticks=5,
                                           reflection_cooldown_ticks=10_000)

        async def turn():
            action = await agent._on_tick(_playable(50))
            assert ran == []                                # not before the action is on its way
            await agent._after_turn(_playable(50))
            return action

        action = asyncio.run(turn())
        assert action["action"] == "gather"                 # the model's action, not a stand-in wait
        assert ran == [("tactical", 50)]
        asyncio.run(agent._after_turn(_playable(50)))
        assert ran == [("tactical", 50)]                    # once

    def test_reflection_comes_first_when_both_are_due(self, monkeypatch):
        agent, ran = self._reviewing_agent(monkeypatch, tactical_review_cooldown_ticks=5,
                                           reflection_cooldown_ticks=5)
        asyncio.run(agent._on_tick(_playable(50)))
        asyncio.run(agent._after_turn(_playable(50)))
        assert ran == [("reflection", 50)]

    def test_no_review_in_combat_or_on_an_automatic_wait(self, monkeypatch):
        agent, ran = self._reviewing_agent(monkeypatch, tactical_review_cooldown_ticks=5,
                                           reflection_cooldown_ticks=5)
        for state in (_playable(50, combat_state={"in_combat": True}), _playable(51, travel=_TRAVEL)):
            asyncio.run(agent._on_tick(state))
            asyncio.run(agent._after_turn(state))
        assert ran == []

    def test_a_client_that_cannot_call_back_runs_it_before_returning(self, monkeypatch):
        agent, ran = self._reviewing_agent(monkeypatch, tactical_review_cooldown_ticks=5,
                                           reflection_cooldown_ticks=10_000)

        class _Plain:
            async def run(self, on_tick):
                raise AssertionError("not used")

        agent.client = _Plain()
        action = asyncio.run(agent._on_tick(_playable(50)))
        assert action["action"] == "gather" and ran == [("tactical", 50)]


# ── The client ──────────────────────────────────────────────────────────── #

def test_the_client_calls_back_after_sending_and_can_send_mid_turn(monkeypatch):
    received, order = [], []

    async def handler(ws):
        await ws.recv()
        await ws.send(json.dumps({"type": "auth_ok"}))
        await ws.send(json.dumps({"type": "state", "tick_info": {"current_tick": 3}, "skipped_states": 2}))
        for _ in range(2):
            received.append(json.loads(await ws.recv()))
        await ws.send(json.dumps({"type": "action_queued", "action": "wait", "tick": 3,
                                  "runs_on_tick": 4, "late": True}))
        await ws.close(4001, "done")

    async def main():
        async with serve(handler, "127.0.0.1", 0) as server:
            port = next(iter(server.sockets)).getsockname()[1]
            client = TNEClient(api_key="k", host=f"127.0.0.1:{port}", secure=False, reconnect=False)

            async def on_tick(state):
                assert client.seconds_since_received(state) >= 0
                assert await client.send({"action": "defend"}) is True
                order.append("tick")
                return {"action": "wait"}

            async def after_turn(state):
                order.append(("after", state["tick_info"]["current_tick"]))

            await asyncio.wait_for(client.run(on_tick, after_turn=after_turn), timeout=10)
            assert await client.send({"action": "wait"}) is False       # no connection any more

    asyncio.run(main())
    assert [m["action"] for m in received] == ["defend", "wait"]
    assert order == ["tick", ("after", 3)]


# ── The MCP server ──────────────────────────────────────────────────────── #

class TestMcp:
    def _api(self, monkeypatch, replies):
        calls = []

        def fake_get(url, headers=None, params=None, timeout=None):
            calls.append(("GET", url, params))
            return httpx.Response(200, json=replies[url.rsplit("/", 1)[-1]], request=httpx.Request("GET", url))

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append(("POST", url, json))
            return httpx.Response(202, json={"status": "queued"}, request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx, "get", fake_get)
        monkeypatch.setattr(httpx, "post", fake_post)
        return mcp._GameAPI(api_key="k", host="test", secure=False), calls

    def _call(self, monkeypatch, api, tool, arguments=None):
        sent = []
        monkeypatch.setattr(mcp, "_send_message", sent.append)
        mcp._handle_request("tools/call", {"name": tool, "arguments": arguments or {}}, 1, api)
        return sent[0]["result"]

    def test_the_state_is_lean_and_compact_by_default(self, monkeypatch):
        api, calls = self._api(monkeypatch, {"state": {"format": 2, "tick": 12, "self": {"level": 3}}})
        result = self._call(monkeypatch, api, "get_state")
        assert calls == [("GET", "http://test/v1/agent/state", {"format": 2})]
        assert result["content"][0]["text"] == '{"format":2,"tick":12,"self":{"level":3}}'
        assert api.last_tick == 12

    def test_the_full_state_on_request(self, monkeypatch):
        api, calls = self._api(monkeypatch, {"state": {"tick_info": {"current_tick": 9}}})
        self._call(monkeypatch, api, "get_state", {"full": True})
        assert calls == [("GET", "http://test/v1/agent/state", None)] and api.last_tick == 9

    def test_an_action_is_marked_with_the_tick_last_read(self, monkeypatch):
        api, calls = self._api(monkeypatch, {"state": {"format": 2, "tick": 12}})
        self._call(monkeypatch, api, "submit_action", {"action": "wait"})
        assert calls[-1][2] == {"action": "wait"}                       # nothing read yet, nothing claimed
        self._call(monkeypatch, api, "get_state")
        self._call(monkeypatch, api, "submit_action",
                   {"action": "wait", "reasoning": "r", "memory": {"plan": "Rest, then trade."}})
        assert calls[-1][2] == {"action": "wait", "reasoning": "r",
                                "memory": {"plan": "Rest, then trade."}, "based_on_tick": 12}

    def test_the_catalogue_in_parts_and_fetched_once(self, monkeypatch):
        catalog = {**_CATALOG, "rules": {"max_staleness_ticks": 1}, "state_guide": {"tick": "…"},
                   "items": {"consumables": {}}}
        api, calls = self._api(monkeypatch, {"catalog": catalog})
        essentials = json.loads(self._call(monkeypatch, api, "get_catalog")["content"][0]["text"])
        assert set(essentials) == {"catalog_version", "game", "actions", "rules", "state_guide"}
        items = json.loads(self._call(monkeypatch, api, "get_catalog", {"section": "items"})["content"][0]["text"])
        assert items == {"catalog_version": "1.0.35+abc", "items": {"consumables": {}}}
        assert len([c for c in calls if c[1].endswith("/catalog")]) == 1
        bad = self._call(monkeypatch, api, "get_catalog", {"section": "secrets"})
        assert bad["isError"] is True and "unknown section" in bad["content"][0]["text"]

    def test_the_tools_are_listed(self):
        names = [tool["name"] for tool in mcp.TOOLS]
        assert names == ["get_state", "get_catalog", "submit_action"]
        assert "memory" in mcp.TOOLS[2]["inputSchema"]["properties"]
