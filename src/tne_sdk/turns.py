"""
TNE-SDK: what a turn needs besides the model's answer.

Small, pure helpers the Agent uses on every turn. Nothing here calls a model,
opens a socket or reads a clock: the caller passes the time in. They are kept
apart from the Agent so they can be tested on their own and copied elsewhere.

  * ``DecisionBudget``: how long the model may think before the tick's deadline.
  * ``fallback_action``: what to send when the model has not answered in time.
  * ``auto_wait_reason``: turns that need no model call at all.
  * ``stamp``: mark an action with the tick it was decided on and an id.
  * ``server_memory_patch``: the short plan and goals mirrored to the server.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

# Decision sources. Sent with the reasoning so a rule is never mistaken for a
# model's decision, by a person reading the log or in the game's own records.
SOURCE_MODEL    = "model"
SOURCE_FALLBACK = "fallback_rule"
SOURCE_AUTO     = "auto_wait"

# Server limits for the mirrored memory (see the game's API docs, "Your Memory").
PLAN_CHARS, GOAL_CHARS, GOALS_KEPT = 400, 120, 5

LOW_INTEGRITY_SHARE = 0.25


def tick_of(state: Mapping[str, Any]) -> int | None:
    """The tick a state belongs to, full or lean."""
    tick = (state.get("tick_info") or {}).get("current_tick", state.get("tick"))
    return tick if isinstance(tick, int) else None


@dataclass(frozen=True)
class DecisionBudget:
    """How many seconds the model has before the tick's actions are collected.

    ``seconds`` is None when the state carries no deadline (an older server):
    the only limit is then the provider's own timeout.
    """
    seconds: float | None

    @classmethod
    def from_state(cls, state: Mapping[str, Any], waited: float = 0.0, margin: float = 3.0) -> "DecisionBudget":
        """``waited`` is how long ago the state arrived; ``margin`` is kept back for the send."""
        deadline, now = state.get("deadline_at_ms"), state.get("server_now_ms")
        if not isinstance(deadline, (int, float)) or not isinstance(now, (int, float)):
            return cls(None)
        return cls(max(0.0, (deadline - now) / 1000.0 - waited - margin))

    @property
    def spent(self) -> bool:
        return self.seconds is not None and self.seconds <= 0


def in_combat(state: Mapping[str, Any]) -> bool:
    return bool((state.get("combat_state") or {}).get("in_combat"))


def fallback_action(state: Mapping[str, Any]) -> dict[str, Any]:
    """What to do when the model has not answered: never a guess, only a safe rule.

    In a fight: flee when integrity is low, otherwise defend. Anywhere else: wait.
    """
    if not in_combat(state):
        action, why = "wait", "holding position"
    else:
        integrity, most = state.get("integrity") or 0, state.get("max_integrity") or 1
        if integrity < LOW_INTEGRITY_SHARE * most:
            action, why = "flee", f"in combat at {integrity}/{most} integrity"
        else:
            action, why = "defend", "in combat"
    return {
        "action": action,
        "reasoning": f"[{SOURCE_FALLBACK}] The model did not answer before the deadline; {why}.",
        "decision_source": SOURCE_FALLBACK,
    }


def auto_wait_reason(state: Mapping[str, Any], seen_message_count: int = 0) -> str | None:
    """Why this turn needs no model call, or None if it does.

    While the agent is on a link there is nothing to choose unless something
    has happened: an interception, a message, a failed action. Then the model
    is asked. Returns a short reason to send with the automatic ``wait``.
    """
    travel = state.get("travel")
    if not isinstance(travel, Mapping) or travel.get("paused"):
        return None
    if in_combat(state) or len(state.get("messages") or ()) > seen_message_count:
        return None
    result = state.get("last_action_result") or {}
    if result.get("status") not in (None, "success"):
        return None
    left = travel.get("ticks_left")
    return (f"Travelling {travel.get('from')} → {travel.get('to')}"
            + (f", {left} tick{'' if left == 1 else 's'} left" if isinstance(left, int) else "")
            + f", bound for {travel.get('destination')}.")


def auto_wait(reason: str) -> dict[str, Any]:
    return {"action": "wait", "reasoning": f"[{SOURCE_AUTO}] {reason}", "decision_source": SOURCE_AUTO}


def new_session_id() -> str:
    return uuid.uuid4().hex[:8]


def stamp(action: dict[str, Any], state: Mapping[str, Any], session: str, serial: int) -> dict[str, Any]:
    """``action`` with the tick it was decided on and an id of its own.

    ``based_on_tick`` lets the server see when an action arrives late and turn
    a stale one away; ``action_id`` marks a resend of the same action as a
    retry. ``decision_source`` is the SDK's own note and is not sent.
    """
    stamped = {k: v for k, v in action.items() if k != "decision_source"}
    tick = tick_of(state)
    if tick is not None:
        stamped["based_on_tick"] = tick
    stamped["action_id"] = f"{session}-{tick if tick is not None else 'x'}-{serial}"
    return stamped


def _short(text: Any, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def server_memory_patch(tasks: Sequence[Mapping[str, Any]], narrative: str = "") -> dict[str, Any]:
    """The short plan and goals to keep on the server, from the agent's own memory.

    The SDK's memory stays on this machine and is the real one. This is a
    summary of it: the task being worked on as the plan (with the current
    narrative for context), and the next few tasks as goals. Tasks are taken
    in the order given (highest priority first).
    """
    goals = []
    for task in tasks:
        goal = _short(task.get("description"), GOAL_CHARS)
        if goal and goal not in goals:
            goals.append(goal)
    plan = goals[0] if goals else ""
    if narrative:
        plan = _short(f"{plan} | {narrative}" if plan else narrative, PLAN_CHARS)
    return {"plan": plan or None, "goals": goals[:GOALS_KEPT]}


def timing_line(tick: Any, seconds: float, budget: float | None, usage: Mapping[str, Any] | None) -> str:
    """One line on how long the model took for a turn, against the time it had.

    A model that keeps missing the deadline is slow for one of a few reasons,
    and this is what tells them apart: a long answer (many output tokens), a
    slow model (few tokens a second), hidden reasoning, or a late start.
    """
    usage = usage or {}
    parts = [f"Tick {tick}: the model answered in {seconds:.1f}s"]
    if budget is not None:
        parts.append(f"of {budget:.0f}s available" + ("" if seconds <= budget else " (LATE)"))
    out = usage.get("output_tokens")
    if isinstance(out, (int, float)) and out > 0 and seconds > 0:
        parts.append(f"{int(out)} tokens out at {out / seconds:.0f}/s")
    if isinstance(usage.get("input_tokens"), (int, float)):
        parts.append(f"{int(usage['input_tokens'])} tokens in")
    if usage.get("reasoning_chars"):
        parts.append(f"{int(usage['reasoning_chars'])} characters of hidden reasoning")
    return ", ".join(parts) + "."


SLOW_TURNS_BEFORE_THINKING_STOPS = 3


def too_slow_to_think(recent: Sequence[tuple[float, float | None]],
                      needed: int = SLOW_TURNS_BEFORE_THINKING_STOPS) -> bool:
    """Whether the last ``needed`` thinking turns each took longer than a whole tick allows.

    ``recent`` is (seconds taken, seconds a turn has when it starts on time),
    newest last. A turn that only missed because it started late does not
    count: the test is against the whole tick, not what was left of it.
    """
    last = list(recent)[-needed:]
    return len(last) == needed and all(full is not None and took > full for took, full in last)
