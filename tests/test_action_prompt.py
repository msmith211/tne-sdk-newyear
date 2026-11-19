"""What the action prompt shows the model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tne_sdk.agent import Agent
from tne_sdk.config import AgentConfig
from tne_sdk.memory.null_memory import NullMemory


def _prompt(state: dict) -> str:
    agent = Agent(AgentConfig(), None, NullMemory(), None)
    with agent.memory:
        return agent._build_action_prompt(state)


def _move(territory: dict) -> dict:
    return {"action": "move", "parameters": {"territory": territory}}


class TestMoveCosts:
    """The server sends move costs as power_cost_per_destination. The prompt
    used to look for a key the server never sends, so the model saw a bare
    list of destinations with no costs."""

    def test_costs_from_the_move_action_are_shown(self):
        prompt = _prompt({"power": 10, "available_actions": [_move({
            "valid_values": ["rust_wastes", "deep_loop"],
            "power_cost_per_destination": {"rust_wastes": 4, "deep_loop": 20},
        })]})
        assert "- rust_wastes (4 power) ✓" in prompt
        assert "- deep_loop (20 power) ✗ not enough power" in prompt

    def test_costs_fall_back_to_the_state(self):
        prompt = _prompt({
            "power": 10, "travel_costs": {"rust_wastes": 4},
            "available_actions": [_move({"valid_values": ["rust_wastes"]})],
        })
        assert "- rust_wastes (4 power) ✓" in prompt

    def test_destinations_without_costs_are_still_listed(self):
        prompt = _prompt({"power": 10, "available_actions": [_move({"valid_values": ["rust_wastes"]})]})
        assert "move(territory): rust_wastes" in prompt


class TestRouteChoice:
    """A Season 4 server sends each journey's ticks and risk, and any other route
    worth weighing. The model has to see them to choose between routes."""

    TERRITORY = {
        "valid_values": ["convergence_point", "rust_wastes"],
        "power_cost_per_destination": {"convergence_point": 11, "rust_wastes": 15},
        "ticks_per_destination": {"convergence_point": 5, "rust_wastes": 1},
        "risk_per_destination": {"convergence_point": 39, "rust_wastes": 33},
        "route_options": {
            "convergence_point": {"quickest": {"power": 13, "ticks": 3, "next_hop": "convergence_point",
                                               "risk": 9, "also": ["safest"]}},
            "rust_wastes": {"safest": {"power": 34, "ticks": 6, "next_hop": "deep_loop", "risk": 19}},
        },
    }

    def test_ticks_and_risk_are_shown_beside_power(self):
        prompt = _prompt({"power": 20, "available_actions": [_move(self.TERRITORY)]})
        assert "- convergence_point (11 power, 5 ticks, 39% risk) ✓" in prompt
        assert "- rust_wastes (15 power, 1 tick, 33% risk) ✓" in prompt

    def test_other_routes_are_listed_under_their_destination(self):
        prompt = _prompt({"power": 20, "available_actions": [_move(self.TERRITORY)]})
        lines = prompt.splitlines()
        at = next(i for i, line in enumerate(lines) if "- convergence_point (" in line)
        assert lines[at + 1].strip() == ('or prefer="quickest" (also the safest): 13 power, 3 ticks, 9% risk, '
                                         'first to convergence_point')
        at = next(i for i, line in enumerate(lines) if "- rust_wastes (" in line)
        assert lines[at + 1].strip() == 'or prefer="safest": 34 power, 6 ticks, 19% risk, first to deep_loop'
        assert '"prefer": "quickest"' in prompt      # how to ask for one

    def test_a_server_that_sends_only_power_is_shown_as_before(self):
        prompt = _prompt({"power": 10, "available_actions": [_move({
            "valid_values": ["rust_wastes"], "power_cost_per_destination": {"rust_wastes": 4}})]})
        assert "- rust_wastes (4 power) ✓" in prompt
        assert 'or prefer=' not in prompt

    def test_malformed_options_are_left_out_not_guessed_at(self):
        territory = dict(self.TERRITORY, route_options={
            "convergence_point": {"quickest": "fast", "scenic": {"power": 1}, "safest": {"ticks": 2}},
            "rust_wastes": ["safest"],
        }, risk_per_destination={"convergence_point": "high", "rust_wastes": True})
        prompt = _prompt({"power": 20, "available_actions": [_move(territory)]})
        assert 'or prefer=' not in prompt
        assert "- convergence_point (11 power, 5 ticks) ✓" in prompt

    def test_the_rules_explain_prefer(self):
        from tne_sdk import prompts
        text = " ".join(str(value) for value in vars(prompts).values() if isinstance(value, str))
        assert 'prefer="quickest"' in text and "cheapest route" in text
