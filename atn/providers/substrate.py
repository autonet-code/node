"""Substrate provider — composite (substrate locate) + (LLM render).

Phase 6.3 of native world-model integration.

The substrate isn't a renderer. It's a **retrieval / working memory**
layer: queries land as coords, the locator finds the related region of
the persistent world, the structure of that region is serialized and
fed to an LLM as context. The LLM does the linguistic synthesis.

  request -> substrate.probe -> rendered region -> LLM(query + region)

This composite provider wraps:
  - A ``WorldService`` (for probing).
  - A "renderer" ``Provider`` (any LLM provider, typically ollama for
    local cheap rendering).

The composite implements the standard ``Provider`` interface so ATN's
agent runtime can use it like any other model.

Phase 6.3 keeps the wrapped renderer pluggable but defaults to nothing
(callers must supply one). Phase 6.4 wires the ollama renderer end-
to-end.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .base import (
    Provider,
    ProviderResponse,
    ToolCall,
    ToolDefinition,
    Usage,
)


log = logging.getLogger(__name__)


_DEFAULT_MAX_REGION_NODES = 16
_DEFAULT_MAX_DISTANCE: Optional[float] = None  # no distance cap by default


_CONTEXT_PROMPT_TEMPLATE = """\
You are answering a query against a knowledge graph called the
substrate. The substrate has located the following region of nodes
relevant to the query. Use this region as the primary source of
truth for your answer; only fall back to your own training when the
region is empty or insufficient.

Each node has a label, a relevance distance (lower = more relevant),
a stake-derived score, and may have a chain of ancestors (parent
nodes that contextualize it).

# Substrate region (most relevant first)
{region_text}

# Query
{query_text}

Answer the query using the region. If the region is empty, answer
from general knowledge but say so explicitly.
"""


class SubstrateProvider(Provider):
    """Composite provider: substrate locate + wrapped LLM render.

    Construction:
      SubstrateProvider(
        world_service=svc,
        renderer=ollama_provider,
        renderer_model="qwen3:4b",  # whatever your ollama has
      )

    Behavior on send():
      1. Extract the most recent user query from messages.
      2. Compute coords via world_service.coords_for_text(query).
      3. probe_inference(coords) → region.
      4. Format the region as text.
      5. Build a system prompt containing the region; pass through
         the original messages list (with the substrate region
         appended to the system prompt) to the wrapped renderer.
      6. Return the renderer's ProviderResponse.

    Phase 6.3 ships the orchestration; Phase 6.4 wires ollama.
    """

    def __init__(
        self,
        world_service: Any,
        renderer: Provider,
        *,
        renderer_model: str = "",
        max_region_nodes: int = _DEFAULT_MAX_REGION_NODES,
        max_distance: Optional[float] = _DEFAULT_MAX_DISTANCE,
    ):
        if world_service is None:
            raise ValueError("SubstrateProvider requires a world_service")
        if renderer is None:
            raise ValueError("SubstrateProvider requires a renderer Provider")
        self._world_service = world_service
        self._renderer = renderer
        self._renderer_model = renderer_model
        self._max_region_nodes = int(max_region_nodes)
        self._max_distance = max_distance

    @property
    def name(self) -> str:
        return "substrate"

    @property
    def renderer(self) -> Provider:
        return self._renderer

    async def send(
        self,
        *,
        messages: List[Dict[str, Any]],
        system: str = "",
        model: str = "",
        max_tokens: int = 1024,
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        # 1. Extract user query.
        query_text = self._extract_user_query(messages)

        # 2-3. Probe the substrate.
        region_summary, render_dict = self._probe(query_text)

        # 4. Format region as text for the LLM.
        region_text = self._format_region_for_prompt(
            region_summary, render_dict,
        )

        # 5. Build augmented system prompt.
        substrate_block = _CONTEXT_PROMPT_TEMPLATE.format(
            region_text=region_text or "(no relevant nodes found)",
            query_text=query_text or "(no query found)",
        )
        # Pre-existing system prompt (if any) wraps the substrate block.
        # The agent-side framing comes first; the substrate context is
        # added below. The renderer sees the whole thing as a system
        # prompt.
        if system:
            augmented_system = f"{system}\n\n{substrate_block}"
        else:
            augmented_system = substrate_block

        # 6. Delegate to the renderer.
        renderer_model = model or self._renderer_model
        response = await self._renderer.send(
            messages=messages,
            system=augmented_system,
            model=renderer_model,
            max_tokens=max_tokens,
            tools=tools,
            temperature=temperature,
        )
        return response

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_user_query(messages: List[Dict[str, Any]]) -> str:
        """The most recent user message becomes the query string.

        Falls back to "" if no user message exists or the content is
        non-textual."""
        for msg in reversed(messages or []):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                # Anthropic-style content blocks: concatenate text parts.
                parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                return "\n".join(p for p in parts if p).strip()
            return str(content).strip()
        return ""

    def _probe(self, query_text: str) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not query_text:
            return [], {}
        coords = self._world_service.coords_for_text(query_text)
        result = self._world_service.probe_inference(
            coords,
            mode="general",
            max_results=self._max_region_nodes,
            max_distance=self._max_distance,
        )
        return result.get("region", []), result.get("render", {})

    @staticmethod
    def _format_region_for_prompt(
        region_summary: List[Dict[str, Any]],
        render_dict: Dict[str, Any],
    ) -> str:
        """Render the substrate region as a plain-text block the LLM
        can consume.

        Keeps it terse: per-node label, distance, score, and (when
        present in the render dict) ancestor chain. The LLM doesn't
        need to know about tendency_id or node_id beyond what helps
        it cite a source.
        """
        if not region_summary:
            return ""

        # Build a per-node ancestor lookup from the render dict.
        # render_dict shape: {"nodes": [{"node_id": ..., "ancestors": [...], "descendants": [...]}, ...]}
        ancestors_by_node: Dict[str, List[Dict[str, Any]]] = {}
        for entry in render_dict.get("nodes", []):
            nid = entry.get("node_id", "")
            if nid:
                ancestors_by_node[nid] = entry.get("ancestors", []) or []

        lines: List[str] = []
        for i, r in enumerate(region_summary, start=1):
            label = r.get("label", "") or r.get("node_id", "")[:12]
            dist = r.get("distance", 0.0)
            score = r.get("score", 0.0)
            line = f"{i}. {label} (distance={dist:.4f}, score={score:.4f})"
            ancestors = ancestors_by_node.get(r.get("node_id", ""), [])
            if ancestors:
                trail = " -> ".join(
                    a.get("content", "")[:60] for a in ancestors if a.get("content")
                )
                if trail:
                    line += f"\n   ancestors: {trail}"
            lines.append(line)
        return "\n".join(lines)
