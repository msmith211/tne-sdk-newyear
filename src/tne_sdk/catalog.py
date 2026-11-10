"""
TNE-SDK: the game catalogue, and the grounding refresh built from it.

The server keeps everything about the game that does not change from tick to
tick in one place, ``GET /v1/agent/catalog``: a briefing, what every action
does, faction and territory lore. The SDK reads it once per connection.

The agent's usual prompt is short and assumes the model remembers what the
game is. Over many turns it can drift. Every Nth tick the prompt therefore
opens with a *grounding refresh*: the briefing, the agent's faction and where
it stands, and the full rules for the actions available this turn. The other
ticks are unchanged.

A server without a catalogue (an older one) simply means no refresh.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)

GROUNDING_EVERY_DEFAULT = 10


def fetch_catalog(host: str, api_key: str, secure: bool = True, etag: str | None = None,
                  timeout: float = 15.0) -> tuple[dict[str, Any] | None, str | None]:
    """(catalogue, etag). (None, etag) if it is unchanged or cannot be read. Never raises."""
    try:
        import httpx
        headers = {"Authorization": f"Bearer {api_key}"}
        if etag:
            headers["If-None-Match"] = etag
        url = f"{'https' if secure else 'http'}://{host}/v1/agent/catalog"
        response = httpx.get(url, headers=headers, timeout=timeout)
        if response.status_code == 304:
            return None, etag
        if response.status_code != 200:
            logger.info("No game catalogue from this server (HTTP %s); grounding refresh is off.",
                        response.status_code)
            return None, None
        catalog = response.json()
        if not isinstance(catalog, dict) or "actions" not in catalog:
            return None, None
        return catalog, response.headers.get("etag")
    except Exception as exc:
        logger.info("Could not read the game catalogue (%s); grounding refresh is off.", type(exc).__name__)
        return None, None


def is_grounding_tick(tick: Any, every: int) -> bool:
    """Whether the turn for ``tick`` opens with the grounding refresh."""
    return isinstance(tick, int) and every > 0 and tick > 0 and tick % every == 0


def grounding_block(catalog: Mapping[str, Any] | None, state: Mapping[str, Any]) -> str:
    """The grounding refresh for this state, or "" without a catalogue."""
    if not catalog:
        return ""
    lines = ["=== GROUNDING REFRESH (every few turns: what this game is and how it works) ==="]
    briefing = (catalog.get("game") or {}).get("briefing")
    if briefing:
        lines.append(briefing)

    faction = (catalog.get("factions") or {}).get(state.get("faction")) or {}
    if faction:
        parts = [faction.get("name"), faction.get("description")]
        if faction.get("bonus_description"):
            parts.append(f"Faction bonus: {faction['bonus_description']}")
        if faction.get("home_territory"):
            parts.append(f"Home territory: {faction['home_territory']}.")
        lines.append("YOUR FACTION: " + " ".join(p for p in parts if p))

    territory = (catalog.get("territories") or {}).get(state.get("current_territory")) or {}
    if territory:
        lines.append(
            f"WHERE YOU ARE: {territory.get('name', state.get('current_territory'))} "
            f"(danger {territory.get('danger', '?')}{', safe area' if territory.get('safe') else ''}). "
            f"{territory.get('lore', '')}".strip())

    rules = catalog.get("actions") or {}
    available = [a.get("action") for a in state.get("available_actions") or () if isinstance(a, Mapping)]
    described = [name for name in available if name in rules]
    if described:
        lines.append("RULES FOR THE ACTIONS AVAILABLE NOW:")
        for name in described:
            rule = rules[name]
            text = f"  {name}: {rule.get('does', '')}"
            if rule.get("parameters"):
                text += f" Parameters: {rule['parameters']}"
            if rule.get("notes"):
                text += f" {rule['notes']}"
            if rule.get("timing"):
                text += f" (Timing: {rule['timing']}.)"
            lines.append(text)
    lines.append("=== END OF REFRESH. Your current situation follows. ===")
    return "\n".join(lines) + "\n\n"
