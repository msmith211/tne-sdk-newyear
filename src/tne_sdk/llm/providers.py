"""
TNE-SDK: LLM Providers

Supports:
- OpenAIProvider           : official OpenAI API
- AnthropicProvider        : Anthropic Claude API
- OpenAICompatibleProvider : any OpenAI-compatible endpoint via direct httpx
  (vLLM, LM Studio, llama.cpp, Ollama, DeepInfra, Together, Groq, etc.)
  Uses httpx directly so all body params (top_k, presence_penalty, etc.) are
  forwarded without the openai SDK stripping unknown fields.
"""
from __future__ import annotations

import inspect
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

import anthropic as _anthropic_sdk
import httpx
import openai as _openai_sdk


@dataclass
class Message:
    """A single message in a chat conversation."""
    role: str     # "system" | "user" | "assistant"
    content: str


class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def chat_completion(
        self,
        messages: list[Message],
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        thinking_mode: bool = False,
        **kwargs: Any,
    ) -> str:
        """Return the assistant response text."""
        ...

    def close(self) -> None:
        """Release any underlying resources. No-op unless overridden."""


DEFAULT_TIMEOUT = 120.0

# The vendor clients retry a failed call by themselves (twice by default), so a
# timeout of T could hold a turn for 3 x T. A turn has one tick to answer in:
# providers here make one attempt, and the agent decides what to do on failure.
_MAX_RETRIES = 0


# Sampling params local servers accept but the official OpenAI API does not.
# Used only if the openai client's create() signature cannot be read.
_NON_OPENAI_PARAMS = frozenset({
    "top_k", "min_p", "repetition_penalty", "chat_template_kwargs",
    "typical_p", "mirostat", "num_ctx",
})


class OpenAIProvider(LLMProvider):
    """LLM Provider for the official OpenAI API."""

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "gpt-4o",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided or found in OPENAI_API_KEY environment variable."
            )
        self._client = _openai_sdk.OpenAI(
            api_key=self.api_key, timeout=timeout, max_retries=_MAX_RETRIES,
        )
        self.default_model = default_model
        self._dropped_params: set[str] = set()
        self._is_supported: Callable[[str], bool] | None = None

    def _supported_check(self) -> Callable[[str], bool]:
        """Build the param check once per provider, on first use."""
        try:
            params = inspect.signature(self._client.chat.completions.create).parameters
        except (TypeError, ValueError):
            return lambda name: name not in _NON_OPENAI_PARAMS
        if any(p.kind is p.VAR_KEYWORD for p in params.values()):
            return lambda name: True
        accepted = frozenset(params)
        return lambda name: name in accepted

    def _supported_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Drop params the openai client would reject.

        The agent always sends local-server params such as top_k. The openai
        client raises TypeError on any keyword create() does not declare, before
        any request is made, so every turn failed.
        """
        if self._is_supported is None:
            self._is_supported = self._supported_check()
        kept: dict[str, Any] = {}
        for name, value in kwargs.items():
            if self._is_supported(name):
                kept[name] = value
            elif name not in self._dropped_params:
                self._dropped_params.add(name)
                logger.warning("OpenAI API does not support %r; not sending it.", name)
        return kept

    def chat_completion(
        self,
        messages: list[Message],
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        thinking_mode: bool = False,
        **kwargs: Any,
    ) -> str:
        model_to_use = model or self.default_model
        response = self._client.chat.completions.create(
            model=model_to_use,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **self._supported_kwargs(kwargs),
        )
        return response.choices[0].message.content or ""


def _usage_of(response: Any) -> dict[str, int]:
    """Token counts from a response, cache reads and writes included. {} if there are none."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    counts = {}
    for name in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
        value = getattr(usage, name, None)
        if isinstance(value, int):
            counts[name] = value
    if counts.get("cache_read_input_tokens") or counts.get("cache_creation_input_tokens"):
        logger.debug("Prompt cache: %s tokens read, %s written.",
                     counts.get("cache_read_input_tokens", 0), counts.get("cache_creation_input_tokens", 0))
    return counts


class AnthropicProvider(LLMProvider):
    """LLM Provider for the Anthropic Claude API."""

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "claude-sonnet-4-6",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or ""
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not provided or found in ANTHROPIC_API_KEY environment variable."
            )
        self._client = _anthropic_sdk.Anthropic(
            api_key=self.api_key, timeout=timeout, max_retries=_MAX_RETRIES,
        )
        self.default_model = default_model
        self.last_usage: dict[str, int] = {}    # token counts of the last call

    def chat_completion(
        self,
        messages: list[Message],
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        thinking_mode: bool = False,
        **kwargs: Any,
    ) -> str:
        model_to_use = model or self.default_model

        # Anthropic uses a top-level system param; strip unsupported kwargs
        cache_system = kwargs.pop("cache_system", True)
        system_prompt = ""
        filtered: list[dict[str, str]] = []
        for m in messages:
            if m.role == "system":
                system_prompt = m.content
            else:
                filtered.append({"role": m.role, "content": m.content})

        # Anthropic does not support these; strip silently
        for unsupported in ("top_p", "presence_penalty", "chat_template_kwargs",
                            "top_k", "frequency_penalty"):
            kwargs.pop(unsupported, None)

        # The system prompt is the same on every turn, so it is marked as a
        # cache point: after the first call it is read back at a fraction of
        # the input price while calls come less than five minutes apart. A
        # prompt shorter than the model's minimum is simply not cached.
        system: Any = system_prompt
        if cache_system and system_prompt:
            system = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]

        response = self._client.messages.create(
            model=model_to_use,
            system=system,
            messages=filtered,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        self.last_usage = _usage_of(response)
        return response.content[0].text


class OpenAICompatibleProvider(LLMProvider):
    """
    LLM Provider for any OpenAI-compatible endpoint.

    Uses direct httpx POST so all body params (top_k, presence_penalty,
    chat_template_kwargs, etc.) are forwarded without stripping by the
    openai SDK.  Ideal for vLLM, LM Studio, llama.cpp, Ollama, DeepInfra,
    OpenRouter, Together, Groq, etc.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "not-needed",
        default_model: str = "local-model",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must be provided for OpenAICompatibleProvider.")
        # Normalise: strip trailing slash, ensure /chat/completions is appended later
        self._completions_url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self.default_model = default_model
        self._timeout = timeout
        self._http = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        self.last_usage: dict[str, int] = {}    # token counts of the last call
        # Thinking is switched off through chat_template_kwargs, which not every
        # server reads. When a model reasons anyway, the next calls also ask with
        # ``reasoning_effort``; a server that refuses that is not asked again.
        self._reasoned_when_off = False
        self._effort_switch_refused = False
        self._told_still_reasoning = False

    def chat_completion(
        self,
        messages: list[Message],
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        thinking_mode: bool = False,
        **kwargs: Any,
    ) -> str:
        model_to_use = model or self.default_model

        body: dict[str, Any] = {
            "model":       model_to_use,
            "messages":    [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "top_p":       top_p,
        }

        # Forward all extra params directly (top_k, presence_penalty, etc.)
        body.update(kwargs)

        # Thinking toggle via chat_template_kwargs (Qwen3, Qwen3.5, vLLM, SGLang)
        body.setdefault("chat_template_kwargs", {})["enable_thinking"] = thinking_mode

        effort_switch = (
            not thinking_mode and self._reasoned_when_off and not self._effort_switch_refused
            and "reasoning_effort" not in body
        )
        if effort_switch:
            body["reasoning_effort"] = "none"
        response = self._http.post(self._completions_url, json=body)
        if effort_switch and response.status_code in (400, 422):
            del body["reasoning_effort"]
            again = self._http.post(self._completions_url, json=body)
            if again.status_code < 400:
                # Only now is it known that the field was the problem. A 400
                # for another reason fails again here and is raised below.
                self._effort_switch_refused = True
                logger.warning(
                    "This server does not accept reasoning_effort, so the model's thinking cannot be "
                    "switched off from here. Switch it off in the server's own settings for this model.")
            response = again
        response.raise_for_status()
        data = response.json()
        msg = data["choices"][0]["message"]
        if not thinking_mode and (msg.get("reasoning_content") or "<think>" in (msg.get("content") or "")):
            if not self._reasoned_when_off:
                self._reasoned_when_off = True
                logger.warning(
                    "Thinking is off, but the model reasoned before answering, which makes every "
                    "turn slower. Asking again with reasoning_effort from the next call.")
            elif effort_switch and not self._told_still_reasoning:
                self._told_still_reasoning = True
                logger.warning(
                    "The model still reasons with thinking switched off both ways. Switch it off in "
                    "the server's own settings for this model, or set it in default_llm_kwargs.")
        usage = data.get("usage") or {}
        self.last_usage = {
            ours: int(usage[theirs])
            for ours, theirs in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens"))
            if isinstance(usage.get(theirs), (int, float))
        }
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict) and isinstance(details.get("reasoning_tokens"), (int, float)):
            self.last_usage["reasoning_tokens"] = int(details["reasoning_tokens"])
        if msg.get("reasoning_content"):
            self.last_usage["reasoning_chars"] = len(msg["reasoning_content"])

        # Log reasoning_content if the server separates it (--reasoning-format auto).
        # This doesn't affect the return value - just makes it visible in logs.
        reasoning = msg.get("reasoning_content")
        if reasoning:
            logger.debug(
                "Model reasoning (%d chars): %.200s...", len(reasoning), reasoning
            )

        return msg.get("content") or ""

    def close(self) -> None:
        """Close the underlying httpx client."""
        self._http.close()


# ── Provider factory ──────────────────────────────────────────────────────── #

def provider_from_profile(profile: dict[str, Any], timeout: float = DEFAULT_TIMEOUT) -> LLMProvider:
    """
    Auto-select the right LLM provider from a profile dict.

    Selection rules (based on llm_url):
      - "bedrock://"    -> BedrockProvider    (AWS Bedrock Converse API)
      - "anthropic.com" -> AnthropicProvider  (native Anthropic API)
      - "openai.com"    -> OpenAIProvider      (official OpenAI SDK)
      - anything else   -> OpenAICompatibleProvider (httpx direct, all params forwarded)

    Bedrock profiles use a special URL format:
      ``llm_url: "bedrock://us-east-1"``  (region in the host portion)
    The model field is the Bedrock model ID (e.g. ``"anthropic.claude-sonnet-4-20250514-v1:0"``).

    ``llm_api_key`` is passed through to whichever provider is selected.
    For Anthropic/OpenAI it falls back to the standard env vars
    (ANTHROPIC_API_KEY / OPENAI_API_KEY) if the profile key is empty.
    ``timeout`` (seconds) bounds every LLM call, whichever provider is selected.
    For Bedrock, authentication uses the boto3 credential chain (env vars,
    IAM role, ~/.aws/credentials).
    """
    url   = (profile.get("llm_url") or "").lower()
    key   = profile.get("llm_api_key") or ""
    model = profile.get("model", "local-model")

    if url.startswith("bedrock://"):
        from .bedrock import BedrockProvider
        # Extract region from URL: "bedrock://us-east-1" -> "us-east-1"
        region = url.replace("bedrock://", "").strip("/") or "us-east-1"
        return BedrockProvider(
            model_id          = model,
            region            = region,
            default_max_tokens = int(profile.get("max_tokens", 1024)),
            timeout           = timeout,
        )
    if "anthropic.com" in url:
        return AnthropicProvider(
            api_key       = key or None,   # falls back to ANTHROPIC_API_KEY env var
            default_model = model,
            timeout       = timeout,
        )
    if "openai.com" in url:
        return OpenAIProvider(
            api_key       = key or None,   # falls back to OPENAI_API_KEY env var
            default_model = model,
            timeout       = timeout,
        )
    return OpenAICompatibleProvider(
        base_url      = profile.get("llm_url", ""),
        api_key       = key or "not-needed",
        default_model = model,
        timeout       = timeout,
    )
