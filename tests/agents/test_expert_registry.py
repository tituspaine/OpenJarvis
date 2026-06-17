"""Tests for the faithful ToolOrchestra unified-tool registry."""

from __future__ import annotations

import random

import pytest

from openjarvis.agents.hybrid.expert_registry import (
    ExpertTool,
    KIND_MODEL,
    build_tool_specs,
    default_catalog,
    sample_tool_config,
    to_worker_dict,
    tools_by_name,
)


def test_each_model_is_its_own_tool():
    """Faithful §3.1: one named tool per model, not a meta-tool + slot."""
    cat = default_catalog()
    names = {t.name for t in cat}
    # Distinct model tools, each with its own name.
    for n in ("gpt_5", "gpt_5_mini", "qwen3_32b", "qwen2_5_coder_32b",
              "llama_3_3_70b", "claude_opus"):
        assert n in names, f"missing model tool {n}"
    # No meta-tool / slot vocabulary leaks in.
    assert "answer" not in names and "enhance_reasoning" not in names


def test_catalog_names_unique_and_valid():
    cat = default_catalog()
    names = [t.name for t in cat]
    assert len(names) == len(set(names))
    assert all(isinstance(t, ExpertTool) for t in cat)


def test_local_model_included_only_when_served():
    assert "local_model" not in {t.name for t in default_catalog()}
    cat = default_catalog(local_model="qwen3:8b", local_endpoint="http://x/v1")
    local = tools_by_name(cat)["local_model"]
    assert local.backend_type == "vllm"
    assert local.base_url == "http://x/v1"
    assert local.price_in == 0.0 and local.price_out == 0.0


def test_invalid_tool_rejected():
    with pytest.raises(ValueError):
        ExpertTool(name="x", kind="bogus", backend_type="openai", summary="", model="m")
    with pytest.raises(ValueError):
        ExpertTool(name="x", kind=KIND_MODEL, backend_type="openai", summary="", model=None)


def test_specs_shape_and_pricing_in_description():
    cat = default_catalog()
    specs = build_tool_specs(cat)
    by = {s["function"]["name"]: s for s in specs}
    gpt5 = by["gpt_5"]
    assert gpt5["type"] == "function"
    assert "input" in gpt5["function"]["parameters"]["properties"]
    # Price table is surfaced in the description (the policy is trained on it).
    assert "$1.25/1M input" in gpt5["function"]["description"]
    # Search tool takes a query, code takes code.
    assert "query" in by["web_search"]["function"]["parameters"]["properties"]
    assert "code" in by["code_interpreter"]["function"]["parameters"]["properties"]


def test_sample_is_deterministic_and_well_formed():
    cat = default_catalog(local_model="qwen3:8b", local_endpoint="http://x/v1")
    a = sample_tool_config(cat, rng=random.Random(0), min_tools=4)
    b = sample_tool_config(cat, rng=random.Random(0), min_tools=4)
    assert [t.name for t in a] == [t.name for t in b]  # deterministic
    assert len(a) >= 4
    assert any(t.kind == KIND_MODEL for t in a)         # can reason
    assert any(t.kind != KIND_MODEL for t in a)         # can act
    assert {t.name for t in a} <= {t.name for t in cat}  # subset


def test_price_jitter_changes_prices_reproducibly():
    cat = default_catalog()
    base = {t.name: t for t in sample_tool_config(cat, rng=random.Random(3), min_tools=8)}
    jit = {t.name: t for t in sample_tool_config(
        cat, rng=random.Random(3), min_tools=8, price_jitter=0.5)}
    # Same subset (same seed/sequence up to jitter draws), but model prices move.
    moved = [n for n in base
             if base[n].kind == KIND_MODEL and base[n].price_in
             and n in jit and jit[n].price_in != base[n].price_in]
    assert moved, "expected jitter to change at least one model price"
    for n in moved:
        f = jit[n].price_in / base[n].price_in
        assert 0.5 <= f <= 1.5


def test_to_worker_dict_maps_backend():
    cat = default_catalog(local_model="qwen3:8b", local_endpoint="http://x/v1")
    by = tools_by_name(cat)
    assert to_worker_dict(by["gpt_5"]) == {
        "name": "gpt_5", "type": "openai", "model": "gpt-5"}
    local = to_worker_dict(by["local_model"])
    assert local["type"] == "vllm" and local["base_url"] == "http://x/v1"
