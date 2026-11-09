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
