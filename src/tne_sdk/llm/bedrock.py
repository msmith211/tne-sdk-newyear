"""
TNE-SDK: AWS Bedrock LLM Provider

Uses the Bedrock Converse API - a unified interface that works across all
Bedrock-hosted models (Anthropic, Meta, Mistral, Cohere, Amazon Nova,
Google Gemma, Qwen, Nvidia, etc.) without per-model request/response
schema handling.

Requires ``boto3``:  pip install "tne-sdk[bedrock]"

Authentication uses the standard boto3 credential chain:
  1. Explicit credentials via constructor (access_key / secret_key)
  2. Environment variables (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
  3. Shared credentials file (~/.aws/credentials)
  4. IAM role (EC2 instance profile, ECS task role, Lambda execution role)

For most server deployments, IAM roles are recommended - no keys to manage.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from .providers import LLMProvider, Message

logger = logging.getLogger(__name__)


# Model families that accept a prompt cache point in the Converse API. A model
# from one of these that still refuses is remembered and not asked again.
_CACHING_MODEL_FAMILIES = ("anthropic.claude", "amazon.nova")


class BedrockProvider(LLMProvider):
    """
    LLM Provider for AWS Bedrock via the Converse API.

    Parameters
    ----------
    model_id : str
        Bedrock model identifier, e.g. ``"anthropic.claude-sonnet-4-20250514-v1:0"``,
        ``"mistral.ministral-3-8b-instruct"``, ``"qwen.qwen3-32b-v1:0"``.
    region : str
        AWS region for the Bedrock endpoint (default: ``"us-east-1"``).
    access_key : str | None
        Explicit AWS access key. If None, uses the default boto3 credential chain.
    secret_key : str | None
        Explicit AWS secret key. If None, uses the default boto3 credential chain.
    session_token : str | None
        Optional session token for temporary credentials (STS).
    default_max_tokens : int
        Default max tokens if not overridden per call.
    timeout : float
        Seconds to wait for a response. One attempt is made: boto3's own
        retries are off, so a call never takes longer than this.
    """

    def __init__(
        self,
        model_id: str,
        region: str = "us-east-1",
        access_key: str | None = None,
        secret_key: str | None = None,
        session_token: str | None = None,
        default_max_tokens: int = 1024,
        timeout: float = 120.0,
    ) -> None:
        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            raise ImportError(
                "boto3 is required for BedrockProvider.\n"
                "  Install with: pip install \"tne-sdk[bedrock]\"\n"
                "  Or directly:  pip install boto3"
            )

        if not model_id:
            raise ValueError("model_id is required for BedrockProvider.")

        self.model_id = model_id
        self.default_max_tokens = default_max_tokens

        # Build client kwargs - only include credentials if explicitly provided.
        # Otherwise boto3 uses its standard credential chain (env vars, IAM role, etc.)
        client_kwargs: dict[str, Any] = {
            "region_name": region,
            "config": Config(
                read_timeout=timeout,
                connect_timeout=min(timeout, 10.0),
                retries={"total_max_attempts": 1},
            ),
        }
        if access_key and secret_key:
            client_kwargs["aws_access_key_id"] = access_key
            client_kwargs["aws_secret_access_key"] = secret_key
            if session_token:
                client_kwargs["aws_session_token"] = session_token

        self._client = boto3.client("bedrock-runtime", **client_kwargs)
        self._no_cache_point: set[str] = set()   # models that refused a prompt cache point
        self.last_usage: dict[str, int] = {}     # token counts of the last call

    def _caches_prompts(self, model_id: str) -> bool:
        """Whether to send a cache point for this model: the model families
        that cache prompts on Bedrock, unless this one has already refused."""
        name = model_id.lower()
        return model_id not in self._no_cache_point and any(
            family in name for family in _CACHING_MODEL_FAMILIES)

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
        model_to_use = model or self.model_id

        # Separate system messages from conversation messages.
        # Converse API takes system prompts as a top-level parameter.
        system_blocks: list[dict[str, str]] = []
        conversation: list[dict[str, Any]] = []

        for m in messages:
            if m.role == "system":
                system_blocks.append({"text": m.content})
            else:
                conversation.append({
                    "role": m.role,
                    "content": [{"text": m.content}],
                })

        # Converse requires at least one user message
        if not conversation:
            logger.warning("BedrockProvider: no user/assistant messages provided.")
            return ""

        # Build inference config
        inference_config: dict[str, Any] = {
            "maxTokens": max_tokens or self.default_max_tokens,
            "temperature": temperature,
            "topP": top_p,
        }

        # Build the Converse call
        converse_kwargs: dict[str, Any] = {
            "modelId": model_to_use,
            "messages": conversation,
            "inferenceConfig": inference_config,
        }
        if system_blocks:
            converse_kwargs["system"] = system_blocks
        # The system prompt is the same on every turn. On models that cache
        # prompts, a cache point after it means later calls read it back at a
        # fraction of the input price.
        cached = bool(system_blocks) and self._caches_prompts(model_to_use)
        if cached:
            converse_kwargs["system"] = [*system_blocks, {"cachePoint": {"type": "default"}}]

        # Strip SDK-specific kwargs that Converse doesn't understand
        # (these may be passed through from AgentConfig)
        kwargs.pop("chat_template_kwargs", None)
        kwargs.pop("presence_penalty", None)
        kwargs.pop("top_k", None)
        kwargs.pop("frequency_penalty", None)

        try:
            response = self._client.converse(**converse_kwargs)
        except Exception as exc:
            if not cached or "cache" not in str(exc).lower():
                raise
            # This model or region turned the cache point away. Do without it
            # from now on; the call itself is made again, unchanged otherwise.
            logger.info("Bedrock refused a prompt cache point for %s; continuing without one.", model_to_use)
            self._no_cache_point.add(model_to_use)
            converse_kwargs["system"] = system_blocks
            response = self._client.converse(**converse_kwargs)
        usage = response.get("usage") or {}
        self.last_usage = {name: usage[name] for name in (
            "inputTokens", "outputTokens", "cacheReadInputTokens", "cacheWriteInputTokens",
        ) if isinstance(usage.get(name), int)}

        # Extract text from response content blocks
        content_blocks = response["output"]["message"]["content"]
        text = next((b["text"] for b in content_blocks if "text" in b), None)

        if text is not None:
            return text

        # Fallback: some models with extended thinking may return only a
        # reasoningContent block when max_tokens is too low to fit both
        # reasoning and the final answer. Try to extract from reasoning.
        reasoning_text = next(
            (
                b["reasoningContent"]["reasoningText"]["text"]
                for b in content_blocks
                if "reasoningContent" in b
                and "reasoningText" in b.get("reasoningContent", {})
            ),
            None,
        )
        if reasoning_text:
            logger.warning(
                "Bedrock returned reasoning-only response for %s - "
                "max_tokens may be too low for thinking + output.",
                model_to_use,
            )
            return reasoning_text

        logger.warning(
            "No text content in Bedrock response for %s: %s",
            model_to_use,
            content_blocks,
        )
        return ""
