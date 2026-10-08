from __future__ import annotations

import pytest

from preauth.llm.base import LLMError
from preauth.llm.cascade import CascadeLLM
from preauth.llm.factory import build_llm
from preauth.llm.fake import FailingLLM, FakeLLM
from preauth.llm.providers import OpenAIClient, build_client


def test_cascade_falls_back():
    cascade = CascadeLLM([FailingLLM("a"), FailingLLM("b"), FakeLLM("ok")])
    result = cascade.complete("s", "p")
    assert result.text == "ok"


def test_cascade_all_fail_raises():
    with pytest.raises(LLMError):
        CascadeLLM([FailingLLM("a"), FailingLLM("b")]).complete("s", "p")


def test_cascade_requires_clients():
    with pytest.raises(ValueError):
        CascadeLLM([])


def test_cascade_model_name():
    assert CascadeLLM([FakeLLM(), FailingLLM("x")]).model == "cascade[fake,x]"


def test_build_llm_single_and_cascade():
    assert build_llm("openai").name == "openai"
    multi = build_llm("anthropic,openai,ollama")
    assert isinstance(multi, CascadeLLM)
    assert [c.name for c in multi.clients] == ["anthropic", "openai", "ollama"]


def test_build_client_unknown():
    with pytest.raises(ValueError):
        build_client("nope")


def test_provider_without_key_raises():
    with pytest.raises(LLMError):
        OpenAIClient(api_key="").complete("s", "p")
