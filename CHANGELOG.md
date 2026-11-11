# Changelog

All notable changes to `tne-sdk` are listed here. Release notes for each
version are also published on the
[GitHub Releases page](https://github.com/Firespawn-Studios/tne-sdk/releases).

## Unreleased

Season 4 work. Not published.

### Changed

- **Sampled prompts leave out what other agents wrote.** A prompt sent in
  `client_report` for the research dataset keeps each message's sender and
  drops its text, and drops the description of any event that quotes a
  message. The model is still shown the real prompt.

### Fixed

- **Found in the final review of the Season 4 work.**
  - Bedrock token counts are reported under the same names as the other
    providers, so they reach the timing line and the research report.
  - A model call that fails is reported as `model_error` with no token counts.
    It used to carry the counts of the previous call, often a reflection.
    A retry after bad JSON now adds both calls' counts up.
  - Logging a state's events can no longer stop the agent answering.
  - The memory mirror is sent every turn until the server's copy matches. It
    used to be marked as sent even when the action carrying it was turned away.
  - A 400 from the model server for another reason no longer makes the SDK
    give up on `reasoning_effort` for the rest of the run.
  - A late action is noted as soon as it is sent, so the next turn's fallback
    cannot replace it before the server's confirmation is read.
  - The MCP server reads the catalogue again when the game says it has changed.

### Added

- **Research reporting, when the game asks.** A state that carries a
  `research` object (the game's own agents, in a deployment that records
  research data) gets a `client_report` on every action: SDK version, model,
  prompt-template hash, who decided (the model, the fallback rule or an
  automatic wait), token counts and how long the call took. The prompt itself
  is added only on the turns the server samples. Without that object nothing
  is reported.
- **Slow local models.** Each turn logs how long the model took against the
  time it had, with tokens out, tokens a second and any hidden reasoning. With
  thinking on, `action_thinking: auto` (the default) stops thinking on action
  turns once three in a row each take longer than a whole tick; reviews keep
  thinking. `on` and `off` fix it either way. A local server that reasons with
  thinking off is asked again with `reasoning_effort`. The deadline fallback
  no longer replaces the agent's own late action queued for the same tick.
- **Actions say which state they were decided on.** Every action carries
  `based_on_tick` and an `action_id`. A server that understands them runs a
  late action only if it is still safe to, and turns a stale one away; the
  model is told when that happens and decides again. Older servers ignore
  both fields.
- **A fallback at the deadline.** If the model has not answered when the
  tick's actions are about to be collected, the agent sends a safe rule
  (`wait`; `defend`, or `flee` at low integrity, in combat), marked
  `[fallback_rule]` so it is never mistaken for the model's decision. The
  model is not cut off: its answer is sent when it arrives. Counted in
  `Agent.fallbacks_sent`. Turn off with `deadline_fallback: false`.
- **No model call while travelling.** On a link with nothing new (no
  interception, no message, no failed action) the agent sends `wait` itself,
  marked `[auto_wait]`. Counted in `Agent.auto_waits`. Turn off with
  `auto_wait_travel: false`.
- **A grounding refresh every tenth tick.** The prompt opens with the game
  briefing, your faction, where you are, and the full rules for the actions
  available that turn, read from the server's catalogue. The other nine ticks
  are unchanged. Set the interval with `grounding_every_ticks` (`0` for off).
  A server without a catalogue means no refresh.
- **A short plan and goals kept on the server.** The agent mirrors its current
  task and next goals to the server's memory fields when they change. Local
  memory is unchanged and stays the full record. Turn off with
  `mirror_memory: false`.
- **`tne-mcp`: lean state, catalogue tool.** `get_state` returns the lean
  state as compact JSON by default (`full: true` for the full one), and a new
  `get_catalog` tool returns the briefing and action rules. `submit_action`
  takes an optional `memory` object and marks each action with the tick of
  the state last read.
- **Prompt caching for Anthropic and Bedrock.** The system prompt is the same
  on every turn, so it is now sent as a cache point. After the first call it
  is read back at a fraction of the input price while calls come less than
  five minutes apart. Anthropic: on by default (`cache_system=False` on a call
  turns it off). Bedrock: sent for Claude and Nova models; a model that
  refuses the cache point is remembered and called without it. A prompt
  shorter than the model's minimum cacheable length is not cached; nothing
  fails. Token counts, including cache reads and writes, are on
  `provider.last_usage`. OpenAI caches a repeated prefix by itself and needs
  no change.
- `TNEClient.run` takes `after_turn`, called once the turn's action is sent,
  and `TNEClient.send` sends an action from inside a turn.

### Changed

- **Reflection and tactical review no longer cost a turn.** They used to run
  in place of an action, so the agent sent `wait` one tick in ten. They now
  run after the turn's action has been sent.

### Fixed

- **Slow turns no longer leave the agent behind.** If a turn took longer than
  a tick, the states that arrived meanwhile queued up and the agent answered
  each one in order, acting on facts that were already out of date and never
  catching up. The client now keeps only the newest state: the next turn acts
  on it, and the skipped ones are logged and counted in
  `TNEClient.states_skipped`.
- **`llm_timeout` applies to every provider.** OpenAI, Anthropic and Bedrock
  profiles ignored it. They also retried failed calls on their own, so one
  slow call could hold a turn for several minutes. Every provider now makes
  one attempt, bounded by `llm_timeout`; a failed call is a `wait` for that
  tick, as before.
- **Move costs in the prompt.** The action list showed destinations without
  their power costs, because it looked for a field the server doesn't send.

## [1.0.2] - 2026-10-29

A reliability release. OpenAI works again, agents stay connected through
server restarts and network drops, and actions match the stricter checks in
the latest server update.

**Upgrade:** `pip install -U tne-sdk`. No config or code changes are needed.

### Fixed

- **OpenAI.** Profiles using `https://api.openai.com/v1` never made a
  successful call; every turn fell back to `wait`. They work now. Settings the
  OpenAI API doesn't support, such as `top_k`, are skipped, with a one-time
  warning in the log.
- **Reconnecting.** The client now reconnects after every disconnect except an
  invalid API key or a dormant agent. Before, some clean disconnects stopped it
  for good. After a long healthy session, the first reconnect also happens
  within seconds instead of waiting up to a minute.
- **Stalled connections.** If the connection stays open but game state stops
  arriving for three ticks (about three minutes), the client reconnects instead
  of waiting forever. Time your agent spends thinking doesn't count, so slow
  models are not affected.
- **Actions and the latest server update.** The server now rejects an action if
  a parameter holds an unexpected object. When a model adds extra fields to its
  reply, such as `"thoughts": {...}`, the agent drops them instead of losing the
  turn. A malformed `parameters` value is treated as empty instead of failing
  the turn.
- **Unreadable server messages** are skipped instead of stopping the client.
- **`tne-mcp`.** When the server refuses an action, your assistant now sees the
  server's reason, such as the list of valid actions, instead of just
  `400 Bad Request`.
- **`tne-relay`:**
  - It gets the same reconnect and stall fixes.
  - `relay/result.json` now reports refusals as
    `{"status": "error", "code": ..., "message": ...}`.
  - An `action.json` that can't be read is reported there too. A corrected file
    is still picked up in the same tick.
- **Launcher.** Quitting `tne-launcher` while an agent is running no longer
  crashes or prints errors.

### Changed

- `TNEClient.run()` accepts an optional `on_message` callback, which receives
  every server message other than game state (`action_queued`, `error`, ...).
  Existing code is unaffected.
- Actions sent by the agent now always include a `parameters` object, empty if
  the action takes none. The server treats both forms the same.

### Known issue

- OpenAI reasoning models (the o-series and GPT-5) reject some of the settings
  the agent sends, so they may still return errors. GPT-4o-style models work.
  Support is planned for a later release.

### Compatibility

- Python 3.10–3.13.
- Works with the current game server and the upcoming server update.

## Earlier versions

Versions up to 1.0.1 were published without a changelog. See the
[PyPI release history](https://pypi.org/project/tne-sdk/#history).
