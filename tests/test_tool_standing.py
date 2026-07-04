"""Tool standing — verdict-layer claims for manifests.

Covers WorldService.submit_tool_manifest (two-plane split, dedup on
resubmission, retrieval via infer_artifacts) and manifest_standing on
the live world. Decay tests removed with spec v2 (decay retired —
endpoint offerings are Services now). Design: docs/tool_substrate.md.
"""

from __future__ import annotations

import pytest

from nodes.common.world_service import WorldService
from nodes.common.world_model_substrate.tool_manifest import build_tool_manifest
from nodes.common.world_model_substrate.tool_standing import (
    all_manifest_standings,
    manifest_standing,
)
from nodes.common.world_model_substrate.usefulness_coords import HashingEmbedder

EMBED_DIM = 64


def _svc(tmp_path, label="0xToolSvc"):
    return WorldService(
        label,
        data_root=tmp_path,
        embedding_dim=EMBED_DIM,
        snapshot_every_n_events=10_000,
        snapshot_every_seconds=10_000.0,
    )


def _manifest(**over):
    kwargs = dict(
        name="summarize_csv",
        description="Summarize a CSV file into per-column statistics.",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        author="agent-abc",
        trust_class="pinned",
        code_digest="c" * 64,
        created_ts=1780531200,
    )
    kwargs.update(over)
    return build_tool_manifest(**kwargs)


def _manifest_nodes(svc, digest):
    """Unique claim nodes tagged with the digest. Co-parented nodes show
    up in several tendencies' tree walks — dedup by node id."""
    seen: dict[str, object] = {}
    for tendency in svc._world.tendencies.values():
        for node in tendency.tree.all_nodes():
            if getattr(node, "artifact_digest", "") == digest:
                seen.setdefault(node.id, node)
    return list(seen.values())


class TestSubmitToolManifest:
    def test_two_plane_split(self, tmp_path):
        svc = _svc(tmp_path)
        receipt = svc.submit_tool_manifest(
            _manifest(), agent_id="agent-abc",
            embedder=HashingEmbedder(dim=EMBED_DIM),
        )
        digest = receipt["manifest_digest"]
        assert len(digest) == 64
        # Data plane: full manifest in blob store + index.
        assert svc._blob_store.get_json(digest)["name"] == "summarize_csv"
        assert svc._artifact_index.has(digest)
        # Verdict plane: exactly one claim node tagged with the digest.
        nodes = _manifest_nodes(svc, digest)
        assert len(nodes) == 1
        assert nodes[0].content.startswith("tool summarize_csv:")

    def test_resubmission_dedups(self, tmp_path):
        svc = _svc(tmp_path)
        embedder = HashingEmbedder(dim=EMBED_DIM)
        r1 = svc.submit_tool_manifest(_manifest(), agent_id="a", embedder=embedder)
        r2 = svc.submit_tool_manifest(_manifest(), agent_id="a", embedder=embedder)
        assert r1["manifest_digest"] == r2["manifest_digest"]
        assert len(_manifest_nodes(svc, r1["manifest_digest"])) == 1

    def test_invalid_manifest_rejected(self, tmp_path):
        svc = _svc(tmp_path)
        result = svc.submit_tool_manifest({"kind": "tool_manifest"}, agent_id="a")
        assert "error" in result

    def test_discoverable_via_infer_artifacts(self, tmp_path):
        svc = _svc(tmp_path)
        embedder = HashingEmbedder(dim=EMBED_DIM)
        receipt = svc.submit_tool_manifest(
            _manifest(), agent_id="a", embedder=embedder)
        result = svc.infer_artifacts(
            "summarize_csv Summarize a CSV file into per-column statistics.",
            k=3,
        )
        digests = [a["digest"] for a in result["artifacts"]]
        assert receipt["manifest_digest"] in digests


class TestManifestStanding:
    def test_standing_moves_with_debate(self, tmp_path):
        svc = _svc(tmp_path)
        receipt = svc.submit_tool_manifest(
            _manifest(), agent_id="a", embedder=HashingEmbedder(dim=EMBED_DIM))
        digest = receipt["manifest_digest"]

        base = manifest_standing(svc._world, digest)
        node = _manifest_nodes(svc, digest)[0]
        node.add_post("supporter-1")
        node.add_post("supporter-2")
        assert manifest_standing(svc._world, digest) > base

    def test_all_manifest_standings_sorted(self, tmp_path):
        svc = _svc(tmp_path)
        embedder = HashingEmbedder(dim=EMBED_DIM)
        d1 = svc.submit_tool_manifest(
            _manifest(), agent_id="a", embedder=embedder)["manifest_digest"]
        d2 = svc.submit_tool_manifest(
            _manifest(name="fetch_weather",
                      description="Fetch the weather for a city.",
                      trust_class="attested", code_digest="",
                      connector_id="weather"),
            agent_id="a", embedder=embedder)["manifest_digest"]
        standings = all_manifest_standings(svc._world, [d2, d1, "f" * 64])
        assert list(standings.keys()) == sorted({d1, d2, "f" * 64})
        assert standings["f" * 64] == 0.0  # unknown digest → no claims


