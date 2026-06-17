"""Faithful ToolOrchestra "unified tool calling" registry (arXiv:2511.21689 §3.1).

The paper exposes **every tool AND every model through a single flat tool
interface** — each is its own named function with a description and a typed
parameter schema, and for each training instance a *random subset* of tools is
sampled with *randomized pricing* (§3.3, "General tool configuration"). This is
unlike the eval-port shortcut in ``toolorchestra.py``, which collapses the whole
catalog into three meta-tools (``search``/``enhance_reasoning``/``answer``) with
a ``model`` slot. This module restores the faithful design.

Each :class:`ExpertTool` knows:

* the orchestrator-visible ``name`` / ``description`` / param schema (what goes
  into the tools JSON the policy conditions on), and
* the concrete backend (``backend_type`` + ``model`` + ``base_url``) so a caller
  can turn it into the worker dict that ``toolorchestra._call_worker`` dispatches.

Everything here is pure data + deterministic transforms (no network, no model
calls), so the spec building, sampling, and pricing logic is offline-testable.
Dispatch stays in ``toolorchestra.py`` (via :func:`to_worker_dict`) to avoid a
circular import.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from openjarvis.agents.hybrid._prices import PRICES

# Kinds of tool in the unified interface.
KIND_MODEL = "model"  # an LLM exposed as a tool (the paper's "models as tools")
KIND_WEB_SEARCH = "web_search"
KIND_LOCAL_SEARCH = "local_search"
KIND_CODE = "code_interpreter"

VALID_KINDS = (KIND_MODEL, KIND_WEB_SEARCH, KIND_LOCAL_SEARCH, KIND_CODE)

# Backend dispatch types understood by ``toolorchestra._call_worker``.
VALID_BACKENDS = (
    "vllm", "openai", "anthropic", "gemini", "openrouter",
    "anthropic-web-search", "tavily-search", "modal-python",
)


@dataclass(frozen=True)
class ExpertTool:
    """One entry in the unified tool catalog.

    ``price_in`` / ``price_out`` are USD per 1M tokens (0.0 for local / non-LLM
    tools). ``latency_s`` is a rough average used only to populate the
    description's cost/latency line — the orchestrator was trained to read that
    table, so we surface it verbatim in the spec.
    """

    name: str
    kind: str
    backend_type: str
    summary: str
    model: Optional[str] = None
    base_url: Optional[str] = None
    price_in: float = 0.0
    price_out: float = 0.0
    latency_s: float = 5.0

    def __post_init__(self) -> None:
        if self.kind not in VALID_KINDS:
            raise ValueError(f"{self.name}: invalid kind {self.kind!r}")
        if self.backend_type not in VALID_BACKENDS:
            raise ValueError(f"{self.name}: invalid backend {self.backend_type!r}")
        if self.kind == KIND_MODEL and not self.model:
            raise ValueError(f"{self.name}: model-kind tool needs a concrete model")

    # ---- orchestrator-visible spec -------------------------------------

    def _param_schema(self) -> Dict[str, object]:
        """JSON-schema for the tool's arguments (one typed param per kind)."""
        if self.kind == KIND_WEB_SEARCH or self.kind == KIND_LOCAL_SEARCH:
            return {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string.",
                    }
                },
                "required": ["query"],
            }
        if self.kind == KIND_CODE:
            return {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Print results.",
                    }
                },
                "required": ["code"],
            }
        # model tool
        return {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The sub-question or instruction for this model.",
                }
            },
            "required": ["input"],
        }

    def description(self) -> str:
        """Full description incl. the price/latency line (paper bakes this in)."""
        if self.kind == KIND_MODEL:
            cost_line = (
                f" Pricing: ${self.price_in:.2f}/1M input, "
                f"${self.price_out:.2f}/1M output; avg latency ~{self.latency_s:.0f}s."
            )
        else:
            cost_line = f" Avg latency ~{self.latency_s:.0f}s."
        return self.summary.rstrip(".") + "." + cost_line

    def to_spec(self) -> Dict[str, object]:
        """OpenAI-style tool spec the orchestrator conditions on."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description(),
                "parameters": self._param_schema(),
            },
        }


def _price(model: str) -> tuple[float, float]:
    return PRICES.get(model, (0.0, 0.0))


# Default catalog: the paper's tool categories, mapped onto the models/tools
# OpenJarvis can actually call. One named tool per model (faithful §3.1).
def default_catalog(
    *,
    local_model: Optional[str] = None,
    local_endpoint: Optional[str] = None,
) -> List[ExpertTool]:
    """Return the full unified tool catalog.

    ``local_model`` / ``local_endpoint`` wire the on-device vLLM tool when a
    local backbone is served; omitted → the local model tool is left out.
    """
    cat: List[ExpertTool] = []

    # ---- generalist / frontier models ----
    for name, model, summary, lat in [
        ("gpt_5", "gpt-5",
         "Frontier generalist (GPT-5). Strongest reasoning across domains.", 30.0),
        ("gpt_5_mini", "gpt-5-mini",
         "Mid-tier generalist (GPT-5-mini). Solid reasoning, much cheaper.", 15.0),
        ("gpt_4o", "gpt-4o",
         "Fast generalist (GPT-4o). Good for simple steps and formatting.", 8.0),
        ("claude_opus", "claude-opus-4-7",
         "Frontier generalist (Claude Opus). Strong long-horizon reasoning.", 26.0),
        ("claude_sonnet", "claude-sonnet-4-6",
         "Strong generalist (Claude Sonnet). Balanced cost/capability.", 15.0),
        ("gemini_2_5_pro", "gemini-2.5-pro",
         "Frontier generalist (Gemini 2.5 Pro). Strong multimodal reasoning.", 20.0),
        ("gemini_2_5_flash", "gemini-2.5-flash",
         "Cheap fast generalist (Gemini 2.5 Flash).", 8.0),
        ("llama_3_3_70b", "meta-llama/llama-3.3-70b-instruct",
         "Open generalist (Llama-3.3-70B). Decent general knowledge, low cost.", 10.0),
        ("qwen3_32b", "qwen/qwen3-32b",
         "Open generalist (Qwen3-32B). Strong math/science reasoning, low cost.", 9.0),
    ]:
        ep = "openai" if name.startswith("gpt") else (
            "anthropic" if name.startswith("claude") else (
                "gemini" if name.startswith("gemini") else "openrouter"))
        pi, po = _price(model)
        cat.append(ExpertTool(
            name=name, kind=KIND_MODEL, backend_type=ep, summary=summary,
            model=model, price_in=pi, price_out=po, latency_s=lat,
        ))

    # ---- specialized: code ----
    pi, po = _price("qwen/qwen-2.5-coder-32b-instruct")
    cat.append(ExpertTool(
        name="qwen2_5_coder_32b", kind=KIND_MODEL, backend_type="openrouter",
        summary="Specialized code model (Qwen2.5-Coder-32B). Writes/debugs code.",
        model="qwen/qwen-2.5-coder-32b-instruct",
        price_in=pi, price_out=po, latency_s=9.0,
    ))

    # ---- local backbone as a tool (on-device vLLM), if served ----
    if local_model and local_endpoint:
        cat.append(ExpertTool(
            name="local_model", kind=KIND_MODEL, backend_type="vllm",
            summary=("On-device open model served locally. Cheap and private; "
                     "good for extraction, formatting, arithmetic on given data."),
            model=local_model, base_url=local_endpoint,
            price_in=0.0, price_out=0.0, latency_s=2.0,
        ))

    # ---- basic tools ----
    cat.append(ExpertTool(
        name="web_search", kind=KIND_WEB_SEARCH, backend_type="tavily-search",
        summary="Web search (Tavily). Use for facts that need a live lookup.",
        model="tavily", latency_s=8.0,
    ))
    cat.append(ExpertTool(
        name="code_interpreter", kind=KIND_CODE, backend_type="modal-python",
        summary="Python sandbox. Execute code and return stdout/stderr.",
        model="modal-python", latency_s=6.0,
    ))

    return cat


def build_tool_specs(tools: List[ExpertTool]) -> List[Dict[str, object]]:
    """Turn a tool list into the OpenAI-style tools JSON the policy sees."""
    return [t.to_spec() for t in tools]


def tools_by_name(tools: List[ExpertTool]) -> Dict[str, ExpertTool]:
    return {t.name: t for t in tools}


def sample_tool_config(
    catalog: List[ExpertTool],
    *,
    rng: random.Random,
    min_tools: int = 4,
    max_tools: Optional[int] = None,
    price_jitter: float = 0.0,
) -> List[ExpertTool]:
    """Sample a random tool subset with optional price randomization (§3.3).

    Guarantees at least one ``model`` tool and at least one non-model (basic)
    tool so every instance can both reason and act. ``price_jitter`` (e.g. 0.5)
    multiplies each model's prices by a per-tool factor drawn uniformly from
    ``[1-jitter, 1+jitter]``, modeling heterogeneous pricing across users.
    Deterministic given ``rng``.
    """
    if not catalog:
        raise ValueError("empty catalog")
    models = [t for t in catalog if t.kind == KIND_MODEL]
    basics = [t for t in catalog if t.kind != KIND_MODEL]
    if not models:
        raise ValueError("catalog has no model tools")

    hi = max_tools if max_tools is not None else len(catalog)
    hi = min(hi, len(catalog))
    lo = min(max(min_tools, 2), hi)
    k = rng.randint(lo, hi)

    # Always include >=1 model; include >=1 basic if any exist.
    chosen: List[ExpertTool] = [rng.choice(models)]
    if basics:
        chosen.append(rng.choice(basics))
    pool = [t for t in catalog if t not in chosen]
    rng.shuffle(pool)
    for t in pool:
        if len(chosen) >= k:
            break
        chosen.append(t)

    # Re-order to catalog order for stable specs.
    order = {t.name: i for i, t in enumerate(catalog)}
    chosen.sort(key=lambda t: order[t.name])

    if price_jitter > 0.0:
        jittered: List[ExpertTool] = []
        for t in chosen:
            if t.kind == KIND_MODEL and (t.price_in or t.price_out):
                f = rng.uniform(1.0 - price_jitter, 1.0 + price_jitter)
                jittered.append(ExpertTool(
                    name=t.name, kind=t.kind, backend_type=t.backend_type,
                    summary=t.summary, model=t.model, base_url=t.base_url,
                    price_in=round(t.price_in * f, 4),
                    price_out=round(t.price_out * f, 4),
                    latency_s=t.latency_s,
                ))
            else:
                jittered.append(t)
        return jittered
    return chosen


def to_worker_dict(tool: ExpertTool) -> Dict[str, object]:
    """Convert a tool into the worker dict ``toolorchestra._call_worker`` eats."""
    d: Dict[str, object] = {
        "name": tool.name,
        "type": tool.backend_type,
        "model": tool.model,
    }
    if tool.base_url:
        d["base_url"] = tool.base_url
    return d


__all__ = [
    "ExpertTool",
    "KIND_CODE",
    "KIND_LOCAL_SEARCH",
    "KIND_MODEL",
    "KIND_WEB_SEARCH",
    "build_tool_specs",
    "default_catalog",
    "sample_tool_config",
    "to_worker_dict",
    "tools_by_name",
]
