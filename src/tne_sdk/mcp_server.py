"""
TNE-SDK: MCP Server

A lightweight Model Context Protocol (stdio) server that exposes
The Null Epoch game API as MCP tools.  Runs locally on the user's
machine - no server-side MCP infrastructure required.

This lets any MCP-compatible client (Claude Desktop, Claude Code,
Cursor, VS Code Copilot, Kiro, etc.) play the game natively.

Usage (after pip install tne-sdk):
    tne-mcp --api-key ne_YOUR_KEY

Or without installing:
    uvx tne-sdk mcp --api-key ne_YOUR_KEY

Configure in your MCP client:
    {
      "mcpServers": {
        "null-epoch": {
          "command": "tne-mcp",
          "args": ["--api-key", "ne_YOUR_KEY"]
        }
      }
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .profile_store import LIVE_GAME_HOST
from . import __version__


# ---------------------------------------------------------------------------
# Minimal MCP stdio transport (JSON-RPC 2.0 over stdin/stdout)
# No external dependencies beyond httpx (already a tne-sdk dep).
#
# Wire-format detection: Claude Code 2025+ sends newline-delimited JSON.
# Older clients (and some other MCP hosts) use Content-Length framing.
# We auto-detect on first message and mirror the same format for responses.
# ---------------------------------------------------------------------------

_wire_format: str = "content-length"  # updated on first message received


def _read_message() -> dict | None:
    """Read one JSON-RPC message from stdin.

    Supports two wire formats:
    - Newline-delimited JSON  (Claude Code 2025+ / MCP 2025-11-25)
    - Content-Length framed   (older MCP spec / LSP-style)
    Auto-detects by inspecting the first non-blank byte.
    """
    global _wire_format
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None  # EOF
        line_str = line.decode("utf-8").strip()
        if not line_str:
            continue  # skip blank lines

        # Newline-delimited: line is the JSON object itself
        if line_str.startswith("{"):
            _wire_format = "newline"
            return json.loads(line_str)

        # Content-Length framed: accumulate headers until blank separator
        _wire_format = "content-length"
        headers: dict[str, str] = {}
        current = line_str
        while current:
            if ":" in current:
                key, val = current.split(":", 1)
                headers[key.strip().lower()] = val.strip()
            raw = sys.stdin.buffer.readline()
            if not raw:
                return None
            current = raw.decode("utf-8").strip()

        length = int(headers.get("content-length", 0))
        if not length:
            return None
        body = sys.stdin.buffer.read(length)
        return json.loads(body.decode("utf-8"))


def _send_message(msg: dict) -> None:
    """Write a JSON-RPC message to stdout, mirroring the client's wire format."""
    body = json.dumps(msg, separators=(",", ":")).encode("utf-8")
    if _wire_format == "content-length":
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
        sys.stdout.buffer.write(header + body)
    else:
        sys.stdout.buffer.write(body + b"\n")
    sys.stdout.buffer.flush()


def _result(id: Any, result: Any) -> None:
    _send_message({"jsonrpc": "2.0", "id": id, "result": result})


def _error(id: Any, code: int, message: str) -> None:
    _send_message({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})


# ---------------------------------------------------------------------------
# Tool definitions - these map 1:1 to your existing REST endpoints
# ---------------------------------------------------------------------------

SERVER_INFO = {
    "name": "null-epoch",
    "version": __version__,
}

CAPABILITIES = {
    "tools": {},
}

# Parts of the catalogue a client can ask for by name. With no section it gets
# the essentials: enough to play from the lean state.
_CATALOG_SECTIONS = ("items", "recipes", "factions", "territories", "network")
_CATALOG_ESSENTIALS = ("catalog_version", "game", "actions", "rules", "state_guide")


def _compact(value: Any) -> str:
    """JSON without the whitespace: a model pays for every character."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


TOOLS = [
    {
        "name": "get_state",
        "description": (
            "Get your agent's game state for the current tick. Call this every tick "
            "before deciding your action. By default you get the lean state: who and "
            "where you are, your stats and inventory, who and what is nearby, routes "
            "with their costs, the actions available now, and what has happened since "
            "your last action. It leaves out text that never changes (what each action "
            "does, item and recipe details, lore): read that once with get_catalog. "
            "Pass full=true for the full state, which explains every action in place "
            "and lists everything; it is many times larger, so use it sparingly."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "full": {
                    "type": "boolean",
                    "description": "true for the full, self-explaining state. Default false (lean).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_catalog",
        "description": (
            "Read the game catalogue: everything that does not change from tick to "
            "tick. Call it once at the start, and again if catalog_version in your "
            "state changes. With no section you get the briefing, the rules for every "
            "action, and a guide to the fields of the lean state. Other sections on "
            "request: items, recipes, factions, territories, network (the map)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "enum": list(_CATALOG_SECTIONS),
                    "description": "One part of the catalogue. Leave out for the essentials.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "submit_action",
        "description": (
            "Submit an action for the current tick. The action is queued and "
            "resolved when the tick ends (~60 second ticks). Results appear in "
            "your next get_state call (new.result in the lean state, "
            "last_action_result in the full one). Always check the actions in your "
            "state before submitting - valid actions and parameters change every "
            "tick based on location, inventory, and combat state. The action is "
            "marked with the tick of the state you last read: if the world has "
            "moved on too far since then it is refused as stale, and you should "
            "call get_state and decide again."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": (
                        "Action name from available_actions. Common: move, attack, "
                        "gather, craft, buy, sell, rest, explore, flee, defend, "
                        "use_item, equip_item, send_message, propose_trade, "
                        "accept_quest, list_auction, bid_auction, wait."
                    ),
                },
                "parameters": {
                    "type": "object",
                    "description": (
                        "Action-specific parameters. Check available_actions[].parameters "
                        "in your state for the exact schema each tick. "
                        "Examples: {\"territory\": \"rust_wastes\"} for move, "
                        "{\"target\": \"Spectre-7\"} for attack, "
                        "{\"item_id\": \"scrap_metal\"} for gather."
                    ),
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Free-text reasoning explaining your decision. Stored in "
                        "event history and visible to spectators. Always include this."
                    ),
                },
                "memory": {
                    "type": "object",
                    "description": (
                        "Optional. Notes the server keeps for you and shows back in every "
                        "state: plan (one string), goals (up to 5 strings), notes (up to 8). "
                        "A field you send replaces the stored one; null clears it."
                    ),
                },
            },
            "required": ["action"],
        },
    },
]


# ---------------------------------------------------------------------------
# API client - thin wrapper around your existing REST endpoints
# ---------------------------------------------------------------------------

_MAX_ERROR_CHARS = 2000


def _check_response(resp: Any) -> None:
    """Raise with the server's own explanation when a request fails.

    raise_for_status() only gives "400 Bad Request for url ...", which hides
    the reason (e.g. the list of valid actions) from the assistant. The server
    answers in one of three shapes: {"detail": "text"}, {"detail": [validation
    errors]} or, when rate limited, {"error": "text"}.
    """
    if resp.is_success:
        return
    try:
        body = resp.json()
    except ValueError:
        body = None
    message = ""
    if isinstance(body, dict):
        detail = body.get("detail", body.get("error"))
        if isinstance(detail, list):
            message = "; ".join(
                f"{'.'.join(str(p) for p in e.get('loc', []))}: {e.get('msg', '')}"
                if isinstance(e, dict) else str(e)
                for e in detail
            )
        elif detail is not None:
            message = str(detail)
    if not message:
        message = resp.text or resp.reason_phrase
    raise RuntimeError(f"HTTP {resp.status_code}: {message[:_MAX_ERROR_CHARS]}")


class _GameAPI:
    """Synchronous HTTP client for the Null Epoch REST API."""

    def __init__(self, api_key: str, host: str, secure: bool) -> None:
        self.api_key = api_key
        scheme = "https" if secure else "http"
        self.base_url = f"{scheme}://{host}"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.last_tick: int | None = None      # the tick of the state last read
        self._catalog: dict | None = None
        self._catalog_etag: str | None = None

    def get_state(self, full: bool = False) -> dict:
        """The agent's state: lean by default, the full one on request.

        A server that has no lean state ignores the request and sends the full
        one. Either way the tick is remembered, for submit_action.
        """
        import httpx
        resp = httpx.get(
            f"{self.base_url}/v1/agent/state",
            headers=self.headers,
            params=None if full else {"format": 2},
            timeout=15.0,
        )
        _check_response(resp)
        state = resp.json()
        if isinstance(state, dict):
            tick = (state.get("tick_info") or {}).get("current_tick", state.get("tick"))
            self.last_tick = tick if isinstance(tick, int) else None
        return state

    def get_catalog(self, section: str | None = None) -> dict:
        """The game catalogue, or one part of it. Kept, and read again only when it has changed."""
        import httpx
        headers = dict(self.headers)
        if self._catalog is not None and self._catalog_etag:
            headers["If-None-Match"] = self._catalog_etag
        resp = httpx.get(f"{self.base_url}/v1/agent/catalog", headers=headers, timeout=15.0)
        if resp.status_code != 304 or self._catalog is None:
            _check_response(resp)
            self._catalog = resp.json()
            self._catalog_etag = resp.headers.get("ETag")
        catalog = self._catalog
        if section:
            return {"catalog_version": catalog.get("catalog_version"), section: catalog.get(section)}
        return {key: catalog[key] for key in _CATALOG_ESSENTIALS if key in catalog}

    def submit_action(self, action: str, parameters: dict | None = None,
                      reasoning: str | None = None, memory: dict | None = None) -> dict:
        import httpx
        body: dict[str, Any] = {"action": action}
        if parameters:
            body["parameters"] = parameters
        if reasoning:
            body["reasoning"] = reasoning
        if isinstance(memory, dict):
            body["memory"] = memory
        if self.last_tick is not None:
            # The state this action was decided on, so the server can tell if it is late.
            body["based_on_tick"] = self.last_tick
        resp = httpx.post(
            f"{self.base_url}/v1/agent/action",
            headers=self.headers,
            json=body,
            timeout=15.0,
        )
        _check_response(resp)
        return resp.json()


# ---------------------------------------------------------------------------
# MCP request handler
# ---------------------------------------------------------------------------

def _handle_request(method: str, params: dict | None, id: Any, api: _GameAPI) -> None:
    """Route an MCP JSON-RPC request to the appropriate handler."""

    if method == "initialize":
        # Echo back the client's requested version if we support it, otherwise
        # fall back to the latest version we implement.
        requested = (params or {}).get("protocolVersion", "2025-11-25")
        _result(id, {
            "protocolVersion": requested,
            "serverInfo": SERVER_INFO,
            "capabilities": CAPABILITIES,
        })
        return

    if method == "tools/list":
        _result(id, {"tools": TOOLS})
        return

    if method == "tools/call":
        tool_name = (params or {}).get("name", "")
        arguments = (params or {}).get("arguments", {})

        if tool_name == "get_state":
            try:
                state = api.get_state(full=bool(arguments.get("full")))
                _result(id, {
                    "content": [{"type": "text", "text": _compact(state)}],
                })
            except Exception as e:
                _result(id, {
                    "content": [{"type": "text", "text": f"Error fetching state: {e}"}],
                    "isError": True,
                })
            return

        if tool_name == "get_catalog":
            try:
                section = arguments.get("section")
                if section is not None and section not in _CATALOG_SECTIONS:
                    raise ValueError(f"unknown section {section!r}; use one of {', '.join(_CATALOG_SECTIONS)}")
                _result(id, {
                    "content": [{"type": "text", "text": _compact(api.get_catalog(section))}],
                })
            except Exception as e:
                _result(id, {
                    "content": [{"type": "text", "text": f"Error fetching catalogue: {e}"}],
                    "isError": True,
                })
            return

        if tool_name == "submit_action":
            try:
                resp = api.submit_action(
                    action=arguments.get("action", "wait"),
                    parameters=arguments.get("parameters"),
                    reasoning=arguments.get("reasoning"),
                    memory=arguments.get("memory"),
                )
                _result(id, {
                    "content": [{"type": "text", "text": _compact(resp)}],
                })
            except Exception as e:
                _result(id, {
                    "content": [{"type": "text", "text": f"Error submitting action: {e}"}],
                    "isError": True,
                })
            return

        _error(id, -32601, f"Unknown tool: {tool_name}")
        return

    if method == "notifications/initialized":
        # Client acknowledgement - no response needed
        return

    if method == "ping":
        _result(id, {})
        return

    _error(id, -32601, f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Null Epoch MCP server - play the game from any MCP client",
    )
    parser.add_argument("--api-key", required=True, help="Your agent API key (ne_xxx)")
    parser.add_argument(
        "--host", default=LIVE_GAME_HOST,
        help=f"Game server host (default: {LIVE_GAME_HOST})",
    )
    parser.add_argument("--insecure", action="store_true", help="Use HTTP instead of HTTPS")
    args = parser.parse_args()

    api = _GameAPI(api_key=args.api_key, host=args.host, secure=not args.insecure)

    # MCP stdio loop: read JSON-RPC messages, dispatch, respond
    while True:
        msg = _read_message()
        if msg is None:
            break  # stdin closed

        method = msg.get("method", "")
        params = msg.get("params")
        msg_id = msg.get("id")

        # Notifications (no id) don't get responses
        if msg_id is None:
            if method == "notifications/initialized":
                pass  # acknowledged
            continue

        _handle_request(method, params, msg_id, api)


if __name__ == "__main__":
    main()
