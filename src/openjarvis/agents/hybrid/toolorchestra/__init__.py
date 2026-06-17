"""ToolOrchestraAgent package (split from the former toolorchestra.py module).

Importing this package registers the agent and re-exports the public surface,
so ``from openjarvis.agents.hybrid.toolorchestra import ToolOrchestraAgent``
keeps working unchanged. Submodules:

    prompts   — system prompts, RL tool specs / arg schema
    experts   — slot -> backend worker mapping (default + paper-match)
    sandbox   — Tavily search + Modal Python sandbox helpers
    clients   — orchestrator vLLM tool-call client
    parsing   — action / tool-call parsing + user-prompt assembly
    workers   — worker pool resolution + dispatch (_call_worker etc.)
    agent     — ToolOrchestraAgent (the registered agent class)
"""

from __future__ import annotations

from openjarvis.agents.hybrid.toolorchestra.agent import ToolOrchestraAgent

__all__ = ["ToolOrchestraAgent"]
