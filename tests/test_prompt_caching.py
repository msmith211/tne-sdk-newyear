"""The system prompt is sent as a cache point where the provider caches prompts."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tne_sdk.llm.providers import AnthropicProvider, Message

_MESSAGES = [Message(role="system", content="You are an agent."), Message(role="user", content="Tick 7.")]


def _anthropic(usage=None):
    provider = AnthropicProvider(api_key="sk-ant-test")
    provider._client = MagicMock()
    provider._client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(text='{"action": "wait"}')], usage=usage)
    return provider


class TestAnthropic:
    def test_the_system_prompt_is_a_cache_point(self):
        provider = _anthropic()
        assert provider.chat_completion(messages=_MESSAGES, model="m") == '{"action": "wait"}'
        sent = provider._client.messages.create.call_args.kwargs
        assert sent["system"] == [
            {"type": "text", "text": "You are an agent.", "cache_control": {"type": "ephemeral"}}]
        assert sent["messages"] == [{"role": "user", "content": "Tick 7."}]   # the tick stays outside the cache

    def test_it_can_be_turned_off_per_call(self):
        provider = _anthropic()
        provider.chat_completion(messages=_MESSAGES, model="m", cache_system=False)
        sent = provider._client.messages.create.call_args.kwargs
        assert sent["system"] == "You are an agent." and "cache_system" not in sent

    def test_no_system_prompt_no_cache_point(self):
        provider = _anthropic()
        provider.chat_completion(messages=_MESSAGES[1:], model="m")
        assert provider._client.messages.create.call_args.kwargs["system"] == ""

    def test_cache_reads_and_writes_are_recorded(self):
        usage = SimpleNamespace(input_tokens=310, output_tokens=42,
                                cache_read_input_tokens=1400, cache_creation_input_tokens=0)
        provider = _anthropic(usage)
        provider.chat_completion(messages=_MESSAGES, model="m")
        assert provider.last_usage == {"input_tokens": 310, "output_tokens": 42,
                                       "cache_read_input_tokens": 1400, "cache_creation_input_tokens": 0}

    def test_a_response_without_usage_is_fine(self):
        provider = _anthropic()
        provider.chat_completion(messages=_MESSAGES, model="m")
        assert provider.last_usage == {}


def _bedrock(model_id):
    with patch.dict("sys.modules", {"boto3": MagicMock(), "botocore": MagicMock(),
                                    "botocore.config": MagicMock()}):
        import importlib
        import tne_sdk.llm.bedrock as bedrock
        importlib.reload(bedrock)
        provider = bedrock.BedrockProvider(model_id=model_id)
    provider._client.converse.return_value = {
        "output": {"message": {"content": [{"text": "ok"}]}},
        "usage": {"inputTokens": 300, "outputTokens": 40, "cacheReadInputTokens": 1400},
    }
    return provider


class TestBedrock:
    def test_claude_and_nova_get_a_cache_point_after_the_system_prompt(self):
        for model_id in ("us.anthropic.claude-sonnet-4-6", "amazon.nova-pro-v1:0"):
            provider = _bedrock(model_id)
            assert provider.chat_completion(messages=_MESSAGES) == "ok"
            sent = provider._client.converse.call_args.kwargs
            assert sent["system"] == [{"text": "You are an agent."}, {"cachePoint": {"type": "default"}}]
            assert provider.last_usage == {"input_tokens": 300, "output_tokens": 40, "cache_read_input_tokens": 1400}

    def test_other_models_are_sent_none(self):
        provider = _bedrock("mistral.ministral-3-8b-instruct")
        provider.chat_completion(messages=_MESSAGES)
        assert provider._client.converse.call_args.kwargs["system"] == [{"text": "You are an agent."}]

    def test_a_refused_cache_point_is_dropped_and_not_tried_again(self):
        provider = _bedrock("anthropic.claude-old")
        reply = provider._client.converse.return_value
        provider._client.converse.side_effect = [
            RuntimeError("ValidationException: This model does not support cachePoint"), reply, reply]
        assert provider.chat_completion(messages=_MESSAGES) == "ok"
        first, retry = provider._client.converse.call_args_list
        assert {"cachePoint": {"type": "default"}} in first.kwargs["system"]
        assert retry.kwargs["system"] == [{"text": "You are an agent."}]
        provider.chat_completion(messages=_MESSAGES)
        assert provider._client.converse.call_count == 3          # no cache point, so no second refusal
        assert provider._client.converse.call_args.kwargs["system"] == [{"text": "You are an agent."}]

    def test_any_other_error_is_raised_as_before(self):
        provider = _bedrock("anthropic.claude-sonnet-4-6")
        provider._client.converse.side_effect = RuntimeError("ThrottlingException: slow down")
        try:
            provider.chat_completion(messages=_MESSAGES)
        except RuntimeError as exc:
            assert "Throttling" in str(exc) and provider._client.converse.call_count == 1
        else:
            raise AssertionError("the error was swallowed")
