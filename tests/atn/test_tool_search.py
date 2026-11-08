"""Tool library search (atn/tool_search.py, Decision 2026-11-08): keyword +
semantic over every manifest the daemon holds, merged by reciprocal rank
fusion, relevance order only, metadata (ratings, harm, bond, author,
household) on each match, keyword-only until the model is ready."""
from __future__ import annotations

import json
import random
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from atn import tool_search
from atn.agent_tools import execute_tool
from atn.runtime import review_batch
from atn.tool_search import ToolSearchIndex, bm25_ranking, rrf_merge, search_tools
from atn.tool_store import ToolRecord
from nodes.common.library import minilm

from tests.atn.test_tool_security import (
    AUTH, FakeClient, _make_runtime, _rating, _st,
)
from tests.atn.test_tool_store import _family, _register_echo

SCHEMA = {"type": "object", "properties": {"path": {"type": "string"}}}


def _manifest(name, description, author=AUTH):
    return {"kind": "tool_manifest", "name": name, "description": description,
            "author": author, "trust_class": "pinned", "input_schema": SCHEMA}


R_CSV = "c5" * 32
R_MAIL = "e3" * 32
R_PDF = "f1" * 32
REMOTE = {
    R_CSV: _manifest("csv_parse", "Parse a CSV file into rows of fields."),
    R_MAIL: _manifest("mail_send", "Send an email message over SMTP."),
    R_PDF: _manifest("pdf_text", "Extract the text of a PDF document."),
}


def _with_network(rt, manifests):
    """A library service whose discovery index holds ``manifests`` (remote
    manifests fetched and verified, as _pull_remote_manifests leaves them)."""
    lib = SimpleNamespace(
        artifact_digests=lambda: list(manifests),
        blob_json=lambda d: manifests.get(d),
    )
    rt.autonet = SimpleNamespace(_service=SimpleNamespace(_library=lib))
    return lib


# Concepts the stub model knows: words of one concept embed alike.
_CONCEPTS = {"csv": 0, "spreadsheet": 0, "comma": 0, "rows": 0, "table": 0,
             "email": 1, "mail": 1, "smtp": 1, "inbox": 1,
             "pdf": 2, "document": 2}


class StubModel:
    """Stands in for the loaded MiniLM: a bag of concepts, L2-normalized."""
    dim = minilm.MODEL_DIM

    def embed_batch(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in tool_search.tokenize(t.replace("_", " ")):
                if w in _CONCEPTS:
                    out[i, _CONCEPTS[w]] += 1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


@pytest.fixture
def model():
    minilm.set_shared(StubModel())
    yield
    minilm.set_shared(None)


@pytest.fixture(autouse=True)
def _no_model_leak():
    yield
    minilm.set_shared(None)


# ---------------------------------------------------------------------------
# Keyword search over every manifest held, before the model is ready
# ---------------------------------------------------------------------------

async def test_keyword_search_finds_remote_manifests(tmp_path):
    rt = _make_runtime(tmp_path)
    _with_network(rt, REMOTE)
    out = await search_tools(rt, "parse csv files", 5)
    assert [m["digest"] for m in out["matches"]] == [R_CSV]
    m = out["matches"][0]
    assert m["name"] == "csv_parse" and m["match"]["keyword_rank"] == 1
    assert m["match"]["semantic_rank"] is None
    # Not ready: keyword-only, and the reply says so.
    assert out["semantic"]["ready"] is False
    assert "not ready" in out["semantic"]["note"]


async def test_search_covers_own_tools_and_remote_ones(tmp_path):
    rt = _make_runtime(tmp_path)
    await _family(rt)
    await _register_echo(rt, caller_id="child", name="csv_echo")
    _with_network(rt, REMOTE)
    out = await execute_tool("probe_tools", {"query": "csv", "k": 10}, rt)
    assert {m["name"] for m in out["matches"]} == {"csv_echo", "csv_parse"}
    # A private tool stays scoped to the caller; network ones are public.
    out = await execute_tool("probe_tools", {"query": "csv", "k": 10}, rt,
                             caller_id="sibling")
    assert [m["name"] for m in out["matches"]] == ["csv_parse"]


async def test_semantic_note_carries_download_progress(tmp_path):
    rt = _make_runtime(tmp_path)
    rt.search_model = SimpleNamespace(status=lambda: {
        "state": "downloading", "percent": 34.0,
        "bytes_done": 31 * 2**20, "bytes_total": 86 * 2**20})
    _with_network(rt, REMOTE)
    out = await search_tools(rt, "csv", 3)
    sem = out["semantic"]
    assert sem["ready"] is False and sem["state"] == "downloading"
    assert sem["percent"] == 34.0 and "34%" in sem["note"] and "31/86 MB" in sem["note"]


# ---------------------------------------------------------------------------
# Semantic search once the model is ready; fusion
# ---------------------------------------------------------------------------

async def test_semantic_finds_by_meaning_and_fuses(tmp_path, model):
    rt = _make_runtime(tmp_path)
    _with_network(rt, REMOTE)
    out = await search_tools(rt, "spreadsheet table", 5)
    assert out["semantic"] == {"ready": True}
    assert [m["digest"] for m in out["matches"]] == [R_CSV]
    m = out["matches"][0]["match"]
    assert m["keyword_rank"] is None and m["semantic_rank"] == 1
    assert m["similarity"] > 0.9
    # A query matching both ways ranks the double hit first.
    out = await search_tools(rt, "send mail", 5)
    assert out["matches"][0]["digest"] == R_MAIL
    assert out["matches"][0]["match"]["keyword_rank"] == 1
    assert out["matches"][0]["match"]["semantic_rank"] == 1


def test_rrf_is_deterministic_with_digest_tie_break():
    a, b, c = "aa" * 32, "bb" * 32, "cc" * 32
    # b and c tie: each is first in one list and absent from the other.
    assert rrf_merge([b, a], [c, a]) == rrf_merge([c, a], [b, a])
    merged = rrf_merge([b, a], [c, a])
    assert [d for d, _ in merged] == [a, b, c]
    assert merged[1][1] == merged[2][1]


def test_rank_is_a_pure_function_of_query_and_manifests(tmp_path):
    docs = {f"{i:02x}" * 32: _manifest(f"tool_{i}", f"parse {w} rows")
            for i, w in enumerate(["csv", "csv", "json", "tsv", "csv", "xml"])}
    results = []
    for seed in range(5):
        idx = ToolSearchIndex(None, embed=StubModel().embed_batch,
                              model_key=minilm.model_key)
        order = list(docs)
        random.Random(seed).shuffle(order)
        for d in order:
            idx.note(d, docs[d])
        results.append(idx.rank("csv rows", order))
    assert all(r == results[0] for r in results)
    # Equal keyword scores fall back to the digest.
    csv = [h["digest"] for h in results[0] if h["keyword_rank"]][:3]
    assert csv == sorted(csv)


def test_bm25_prefers_rarer_words_and_name_hits():
    docs = {"a": ["csv", "csv", "parse"], "b": ["parse", "json"], "c": ["mail"]}
    ranked = bm25_ranking("parse csv", docs)
    assert [d for d, _ in ranked] == ["a", "b"]
    assert bm25_ranking("the a of", docs) == []         # stop words only


async def test_order_is_relevance_not_rating(tmp_path, model):
    """A better-rated, less relevant tool does not overtake."""
    rt = _make_runtime(tmp_path)
    weak, strong = "d1" * 32, "d2" * 32
    _with_network(rt, {
        weak: _manifest("csv_rows", "Read csv rows."),
        strong: _manifest("table_tool", "Works on a table."),
    })
    rt.tool_bond_client = FakeClient(
        {weak: _st(), strong: _st()},
        ratings={weak: _rating(correctness=-0.9, harm=0.4),
                 strong: _rating(correctness=0.95)})
    out = await search_tools(rt, "csv rows", 5)
    assert [m["digest"] for m in out["matches"]] == [weak, strong]
    first = out["matches"][0]
    assert first["chain_rating"]["correctness"] == -0.9
    assert first["rating_mass"]["correctness"] == "300"
    assert first["harm"] == 0.4 and first["harm_mass"] == "300"
    assert first["bonded"] is True and first["stake"] == "100"
    assert "author_context" in first
    for gone in ("rating", "axes", "mass", "review_mass"):
        assert gone not in first


async def test_unbonded_tools_stay_excluded(tmp_path):
    rt = _make_runtime(tmp_path)
    _with_network(rt, REMOTE)
    rt.tool_bond_client = FakeClient({R_CSV: _st(bonded=False), R_PDF: _st()})
    out = await search_tools(rt, "csv pdf", 5)
    assert [m["digest"] for m in out["matches"]] == [R_PDF]


# ---------------------------------------------------------------------------
# Embedding lifecycle
# ---------------------------------------------------------------------------

def test_manifests_learned_before_the_model_are_embedded_once_it_is(tmp_path):
    path = tmp_path / "search" / "tool_vectors.jsonl"
    idx = ToolSearchIndex(path)
    for d, m in REMOTE.items():
        idx.note(d, m)
    assert idx.embed_pending() == 0                      # no model yet
    assert idx.pending() == sorted(REMOTE)
    minilm.set_shared(StubModel())
    assert idx.embed_pending() == 3 and idx.pending() == []
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert {r["digest"] for r in rows} == set(REMOTE)
    assert {r["model"] for r in rows} == {minilm.model_key()}
    # A restart loads them: nothing left to embed.
    again = ToolSearchIndex(path)
    for d, m in REMOTE.items():
        again.note(d, m)
    assert again.pending() == []


def test_vectors_from_another_model_are_embedded_again(tmp_path):
    path = tmp_path / "tool_vectors.jsonl"
    idx = ToolSearchIndex(path)
    for d, m in REMOTE.items():
        idx.note(d, m)
    minilm.set_shared(StubModel())
    idx.embed_pending()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["model"] = "sentence-transformers/other-model@abc"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    again = ToolSearchIndex(path)
    for d, m in REMOTE.items():
        again.note(d, m)
    assert again.pending() == [rows[0]["digest"]]
    assert again.embed_pending() == 1
    models = {json.loads(line)["model"] for line in path.read_text().splitlines()}
    assert models == {minilm.model_key()}


async def test_warm_embeds_everything_held(tmp_path, model):
    rt = _make_runtime(tmp_path)
    _with_network(rt, REMOTE)
    tool_search.warm(rt)
    for t in threading.enumerate():
        if t.name == "tool-search-warm":
            t.join(5)
    assert tool_search.index_for(rt).pending() == []


# ---------------------------------------------------------------------------
# The household's own relation to a tool
# ---------------------------------------------------------------------------

def _adopt(rt, digest, manifest):
    rt.tool_store._records[digest] = ToolRecord(
        digest=digest, manifest=manifest, origin="adopted",
        local_author="child")


async def test_household_states(tmp_path):
    rt = _make_runtime(tmp_path)
    await _family(rt)
    _with_network(rt, REMOTE)
    axis_ids = ["correctness", "simplicity"]
    # Adopted and reviewed by us (on chain, in a batch).
    _adopt(rt, R_CSV, REMOTE[R_CSV])
    review_batch.store_for(rt).add(
        "0xagent", 7, axis_ids,
        review_batch.ReviewEntry(digest=R_CSV, axes={"correctness": 0.8}, harm=0.1))
    # Adopted, not reviewed.
    _adopt(rt, R_MAIL, REMOTE[R_MAIL])
    # Reviewed (held until registration), not adopted.
    review_batch.hold_reviews(rt, "child", [
        {"digest": R_PDF, "axes": {"simplicity": -0.5}, "harm": 0.0}])

    out = await search_tools(rt, "csv mail pdf", 5)
    by = {m["digest"]: m["household"] for m in out["matches"]}
    assert by[R_CSV] == {"adopted": True, "authored": False, "reviewed": True,
                         "review": {"axes": {"correctness": 0.8}, "harm": 0.1,
                                    "state": "pending", "epoch": 7}}
    assert by[R_MAIL] == {"adopted": True, "authored": False, "reviewed": False,
                          "review": None}
    assert by[R_PDF]["adopted"] is False and by[R_PDF]["reviewed"] is True
    assert by[R_PDF]["review"] == {"axes": {"simplicity": -0.5}, "harm": 0.0,
                                   "state": "held", "epoch": None}


async def test_household_marks_our_own_tools(tmp_path):
    rt = _make_runtime(tmp_path)
    await _family(rt)
    await _register_echo(rt, caller_id="child", name="csv_echo")
    out = await search_tools(rt, "csv echo", 5)
    assert out["matches"][0]["household"]["authored"] is True
    assert out["matches"][0]["household"]["adopted"] is False


async def test_a_forfeited_batch_does_not_count_as_reviewed(tmp_path):
    rt = _make_runtime(tmp_path)
    _with_network(rt, REMOTE)
    store = review_batch.store_for(rt)
    batch = store.add("0xagent", 3, ["correctness"],
                      review_batch.ReviewEntry(digest=R_CSV, axes={"correctness": 1.0}))
    batch.status = "forfeited"
    store.save(batch)
    out = await search_tools(rt, "csv", 3)
    assert out["matches"][0]["household"]["reviewed"] is False


# ---------------------------------------------------------------------------
# The WS surface
# ---------------------------------------------------------------------------

async def test_ws_probe_and_model_status(tmp_path):
    from atn.ws_server import WebSocketBridge
    rt = _make_runtime(tmp_path)
    _with_network(rt, REMOTE)
    srv = WebSocketBridge.__new__(WebSocketBridge)
    srv.runtime = rt
    owner = SimpleNamespace(scope_ids=None, local=True, owner="0xowner",
                            authed=True, root_agent_id=None)
    r = await srv._handle_message({"type": "probe_tools", "msg_id": "1",
                                   "query": "pdf"}, owner)
    assert r["ok"] and [m["digest"] for m in r["result"]["matches"]] == [R_PDF]
    assert r["result"]["semantic"]["ready"] is False
    r = await srv._handle_message({"type": "search_model_status", "msg_id": "2"}, owner)
    assert r["ok"] and r["result"]["model"] == minilm.MODEL_ID
    assert r["result"]["ready"] is False


# ---------------------------------------------------------------------------
# Fusion weights (Decision 2026-11-09): meaning first, keywords for names
# ---------------------------------------------------------------------------

# Digests chosen so the digest tie-break favors title_case: under equal
# weights the shared word ("title") tied the closer meaning and won on it.
D_TITLE = "0a" * 32
D_SLUG = "fb" * 32
TEXT_SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}}
S0_TEXT_TOOLS = {
    D_TITLE: {**_manifest("title_case", "Capitalize every word of a text."),
              "input_schema": TEXT_SCHEMA},
    D_SLUG: {**_manifest("slugify_text", "Turn a text into a lowercase url slug."),
             "input_schema": TEXT_SCHEMA},
}


class WebAddressModel(StubModel):
    """Ranks slugify_text first for a web-address query: url, slug, web and
    address are one concept; nothing in title_case's text is."""

    def embed_batch(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in tool_search.tokenize(t.replace("_", " ")):
                if w in ("url", "slug", "web", "address", "link"):
                    out[i, 3] += 1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


async def test_meaning_outranks_one_shared_word(tmp_path):
    minilm.set_shared(WebAddressModel())
    rt = _make_runtime(tmp_path)
    _with_network(rt, S0_TEXT_TOOLS)
    out = await search_tools(rt, "make a web address from a title", 5)
    assert out["semantic"] == {"ready": True}
    names = [m["name"] for m in out["matches"]]
    assert names == ["slugify_text", "title_case"], out["matches"]
    slug, title = out["matches"]
    assert slug["match"]["semantic_rank"] == 1 and slug["match"]["keyword_rank"] is None
    assert title["match"]["keyword_rank"] == 1 and title["match"]["semantic_rank"] is None
    assert title["match"]["exact_name"] is False
    assert slug["score"] == pytest.approx(1.0 / 61)
    assert title["score"] == pytest.approx(0.5 / 61)
    # Equal weights tie the two, and the digest put title_case first.
    assert [d for d, _ in rrf_merge([D_TITLE], [D_SLUG])] == [D_TITLE, D_SLUG]


async def test_an_exact_name_takes_keyword_rank_one_at_full_weight(tmp_path):
    rt = _make_runtime(tmp_path)
    d_count, d_essay = "c1" * 32, "c2" * 32
    tools = {
        d_count: _manifest("word_count", "Counts words."),
        # Wins on BM25: it shares "essay" with the query too.
        d_essay: _manifest("essay_stats",
                           "Word count, line count and reading time of an essay."),
    }
    _with_network(rt, tools)
    query = "word count of my essay"
    docs = {d: tool_search.keyword_tokens(m) for d, m in tools.items()}
    assert [d for d, _ in bm25_ranking(query, docs)] == [d_essay, d_count]
    out = await search_tools(rt, query, 5)          # keyword-only: no model
    assert [m["name"] for m in out["matches"]] == ["word_count", "essay_stats"]
    exact, other = out["matches"]
    assert exact["match"] == {"keyword_rank": 1, "semantic_rank": None,
                              "similarity": None, "exact_name": True}
    assert exact["score"] == pytest.approx(1.0 / 61)
    assert other["match"]["keyword_rank"] == 2 and other["match"]["exact_name"] is False
    assert other["score"] == pytest.approx(0.5 / 62)


def test_exact_name_hit():
    hit = tool_search.exact_name_hit
    assert not hit("make a web address from a title", "title_case")
    assert hit("put this in title case", "title_case")
    assert hit("Title-Case it", "title_case")
    assert hit("run slugify-text on it", "slugify-text")
    assert hit("use x.y_tool here", "x.y_tool")       # the full name
    assert not hit("resort the list", "sort")         # words, not substrings
    assert not hit("fix it", "it")                    # stop words only


def test_rrf_weights():
    a, b = "aa" * 32, "bb" * 32
    assert rrf_merge([a], [b], weights=[0.5, 1.0])[0][0] == b
    assert rrf_merge([a], [b], weights=[1.0, 0.5])[0][0] == a
    assert rrf_merge([a], [b]) == rrf_merge([a], [b], weights=[1.0, 1.0])
