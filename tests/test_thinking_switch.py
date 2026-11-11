"""Thinking that a local server leaves on is switched off a second way."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import httpx

from tne_sdk.llm.providers import Message, OpenAICompatibleProvider


def _provider(handler):
    provider = OpenAICompatibleProvider("http://model.test/v1")
    provider._http = httpx.Client(transport=httpx.MockTransport(handler))
    return provider


def _answer(reasoning=None):
    message = {"role": "assistant", "content": '{"action": "wait"}'}
    if reasoning:
        message["reasoning_content"] = reasoning
    return httpx.Response(200, json={"choices": [{"message": message}],
                                     "usage": {"prompt_tokens": 3000, "completion_tokens": 40}})


def _ask(provider, thinking=False):
    return provider.chat_completion([Message(role="user", content="state")], thinking_mode=thinking)


def test_a_model_that_reasons_when_off_is_asked_with_reasoning_effort():
    import json
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        return _answer(None if body.get("reasoning_effort") == "none" else "long thoughts")

    provider = _provider(handler)
    _ask(provider)
    assert "reasoning_effort" not in bodies[0]
    assert provider.last_usage == {"input_tokens": 3000, "output_tokens": 40, "reasoning_chars": 13}
    _ask(provider)
    assert bodies[1]["reasoning_effort"] == "none"
    assert "reasoning_chars" not in provider.last_usage


def test_a_server_that_refuses_the_switch_is_not_asked_again():
    import json
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "reasoning_effort" in body:
            return httpx.Response(400, json={"error": "unknown parameter"})
        return _answer("long thoughts")

    provider = _provider(handler)
    _ask(provider)
    assert _ask(provider) == '{"action": "wait"}'          # refused, then answered without it
    _ask(provider)
    assert ["reasoning_effort" in b for b in bodies] == [False, True, False, False]


def test_thinking_left_on_by_choice_is_left_alone():
    import json
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return _answer("long thoughts")

    provider = _provider(handler)
    _ask(provider, thinking=True)
    _ask(provider, thinking=True)
    assert all("reasoning_effort" not in b for b in bodies)


def test_a_400_for_another_reason_does_not_switch_the_switch_off():
    import json
    bodies, fail = [], {"on": False}

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if fail["on"]:
            return httpx.Response(400, json={"error": "context length exceeded"})
        return _answer(None if body.get("reasoning_effort") == "none" else "long thoughts")

    provider = _provider(handler)
    _ask(provider)                      # reasons: the switch is used from now on
    _ask(provider)
    fail["on"] = True
    try:
        _ask(provider)
        raise AssertionError("the real error must be raised")
    except httpx.HTTPStatusError:
        pass
    fail["on"] = False
    _ask(provider)
    assert bodies[-1].get("reasoning_effort") == "none"       # still asked: the server does accept it
