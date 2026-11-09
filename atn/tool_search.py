"""Tool library search: keyword + semantic, local to this daemon.

Decision 2026-11-08 (docs/tool_substrate.md). Every query runs two searches
over EVERY tool manifest the daemon holds (its own, the ones it adopted,
and every remote manifest it fetched and verified) and merges them:

- **keyword**: BM25 over the manifest's name, description and input-schema
  property names (name words count twice);
- **semantic**: cosine between the query and each manifest embedded with
  the local search model (all-MiniLM-L6-v2, full 384 dims,
  ``nodes/common/library/minilm.py``).

The two ranked lists merge by weighted reciprocal rank fusion (``RRF_K``,
Decision 2026-11-09): the semantic list at ``SEMANTIC_WEIGHT`` (1.0), the
keyword list at ``KEYWORD_WEIGHT`` (0.5), except that an exact-name hit
(``exact_name_hit``) takes keyword rank 1 at full weight. Meaning carries
the order; keywords are there to catch exact names. Ties break on the
digest, so the order is a pure function of the query and the manifests
held. Order is relevance only: the rating (correctness, simplicity,
novelty, the malware share, the charter values and the REP behind them),
bond state, author context and this household's own relation to the tool
come back as metadata on each match, never as a weight. Every tool the
daemon holds can match, unbonded or rated malware included: what to run is
the household's own policy (Decision 2026-11-09).

Until the model is loaded (the first run downloads it), results are keyword
matches and the reply says semantic search is not ready, with the download
progress. Manifests learned before that are embedded once it is. Each stored
vector records the model and revision that produced it; a vector from any
other model is embedded again.

The vectors are daemon-local state (``<data_dir>/search/tool_vectors.jsonl``):
never gossiped, never anchored, never read by the close.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

from nodes.common.library import minilm
from nodes.common.library.tool_manifest import (
    is_tool_manifest,
    manifest_embedding_text,
)

log = logging.getLogger(__name__)

# Reciprocal rank fusion constant (Cormack et al. 2009): each list adds
# weight / (RRF_K + rank) for a manifest it ranks.
RRF_K = 60
# Fusion weights (Decision 2026-11-09). With equal weights a tool sharing
# one word with the query tied a tool matching its meaning; keywords are
# there to catch exact names, which EXACT_NAME_WEIGHT keeps.
SEMANTIC_WEIGHT = 1.0
KEYWORD_WEIGHT = 0.5
EXACT_NAME_WEIGHT = 1.0
# A semantic match below this cosine is not a match (unrelated texts sit
# near 0..0.15 with this model).
SEMANTIC_MIN = 0.2
# BM25 parameters.
BM25_K1 = 1.2
BM25_B = 0.75

_STOP = frozenset("""
a an and are as at be by can do does for from has have how i in into is it
its me my of on or our so that the their them this to up use using want
was we what when which who will with you your
""".split())
_WORD = re.compile(r"[a-z0-9]+")
_NAME_SPLIT = re.compile(r"[_\-\s]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric words, stop words dropped."""
    return [w for w in _WORD.findall(str(text or "").lower()) if w not in _STOP]


def exact_name_hit(query: str, name: str) -> bool:
    """Does ``query`` name the tool? True when every token of ``name``
    (split on ``_``, ``-`` and spaces) is a word of the query, or the query
    contains the full name as a word. A name made of stop words only never
    hits (a tool called ``it`` or ``to_do`` would otherwise top every query
    with those words)."""
    toks = [t for t in _NAME_SPLIT.split(str(name or "").lower()) if t]
    if not toks or all(t in _STOP for t in toks):
        return False
    q = str(query or "").lower()
    if set(toks) <= set(_WORD.findall(q)):
        return True
    full = str(name).lower().strip()
    return re.search(r"(?<![a-z0-9])" + re.escape(full) + r"(?![a-z0-9])",
                     q) is not None


def keyword_tokens(manifest: dict[str, Any]) -> list[str]:
    """The manifest's keyword document: name words twice, then the
    description and the input-schema property names."""
    name = tokenize(str(manifest.get("name") or "").replace("_", " "))
    props = manifest.get("input_schema") or {}
    prop_names: list[str] = []
    if isinstance(props, dict) and isinstance(props.get("properties"), dict):
        prop_names = sorted(props["properties"].keys())
    return (name + name + tokenize(str(manifest.get("description") or ""))
            + tokenize(" ".join(prop_names).replace("_", " ")))


def bm25_ranking(query: str, docs: dict[str, list[str]]) -> list[tuple[str, float]]:
    """``(digest, score)`` for every doc matching a query word, best first,
    ties by digest."""
    q = list(dict.fromkeys(tokenize(query)))
    if not q or not docs:
        return []
    n = len(docs)
    avg = sum(len(t) for t in docs.values()) / n or 1.0
    df = {w: 0 for w in q}
    tfs: dict[str, dict[str, int]] = {}
    for digest, toks in docs.items():
        counts: dict[str, int] = {}
        for t in toks:
            if t in df:
                counts[t] = counts.get(t, 0) + 1
        if counts:
            tfs[digest] = counts
            for t in counts:
                df[t] += 1
    out = []
    for digest in sorted(tfs):
        counts = tfs[digest]
        length = len(docs[digest])
        score = 0.0
        for w in q:
            tf = counts.get(w, 0)
            if not tf:
                continue
            idf = math.log(1.0 + (n - df[w] + 0.5) / (df[w] + 0.5))
            score += idf * tf * (BM25_K1 + 1) / (
                tf + BM25_K1 * (1 - BM25_B + BM25_B * length / avg))
        out.append((digest, round(score, 12)))
    out.sort(key=lambda p: (-p[1], p[0]))
    return out


def rrf_merge(*rankings: list[str], k: int = RRF_K,
              weights: Optional[Iterable[float]] = None) -> list[tuple[str, float]]:
    """Reciprocal rank fusion of ranked digest lists, each list's terms
    times its weight (all 1.0 by default): ``(digest, score)`` best first,
    ties by digest ascending."""
    ws = list(weights) if weights is not None else [1.0] * len(rankings)
    terms: dict[str, list[tuple[float, int]]] = {}
    for w, ranking in zip(ws, rankings):
        for rank, digest in enumerate(ranking, start=1):
            terms.setdefault(digest, []).append((float(w), rank))
    return fuse(terms, k=k)


def fuse(terms: dict[str, list[tuple[float, int]]], *,
         k: int = RRF_K) -> list[tuple[str, float]]:
    """``digest -> [(weight, rank), ...]`` summed as weight / (k + rank):
    ``(digest, score)`` best first, ties by digest ascending."""
    merged = [(d, round(sum(w / (k + r) for w, r in ts), 12))
              for d, ts in terms.items()]
    merged.sort(key=lambda p: (-p[1], p[0]))
    return merged


class ToolSearchIndex:
    """The daemon's search index over tool manifests.

    Keyword documents live in memory (rebuilt from the manifests on every
    start); semantic vectors persist, one row per manifest, tagged with the
    model that produced them.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        *,
        embed: Callable[[list[str]], Optional[np.ndarray]] = minilm.embed_texts,
        model_key: Callable[[], str] = minilm.model_key,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self._embed = embed
        self._model_key = model_key
        self._lock = threading.RLock()
        self._texts: dict[str, str] = {}          # digest -> embedding text
        self._tokens: dict[str, list[str]] = {}   # digest -> keyword doc
        self._names: dict[str, str] = {}          # digest -> manifest name
        self._vectors: dict[str, np.ndarray] = {}
        self._vector_model: dict[str, str] = {}
        self._load()

    # ---- documents ---------------------------------------------------

    def note(self, digest: str, manifest: dict[str, Any]) -> None:
        """Know ``manifest`` (its keyword document now; its vector the next
        time the model is available)."""
        if digest in self._tokens:
            return
        with self._lock:
            self._texts[digest] = manifest_embedding_text(manifest)
            self._tokens[digest] = keyword_tokens(manifest)
            self._names[digest] = str(manifest.get("name") or "")

    def known(self, digest: str) -> bool:
        return digest in self._tokens

    def has_vector(self, digest: str) -> bool:
        return (digest in self._vectors
                and self._vector_model.get(digest) == self._model_key())

    def pending(self) -> list[str]:
        """Known manifests without a vector from the current model."""
        with self._lock:
            return sorted(d for d in self._texts if not self.has_vector(d))

    def embed_pending(self, digests: Optional[Iterable[str]] = None) -> int:
        """Embed the known manifests that lack a current vector (or just
        ``digests``). Returns how many were embedded; 0 while the model is
        not loaded."""
        with self._lock:
            want = sorted(d for d in (digests if digests is not None else self._texts)
                          if d in self._texts and not self.has_vector(d))
            if not want:
                return 0
            vecs = self._embed([self._texts[d] for d in want])
            if vecs is None:
                return 0
            key = self._model_key()
            rows = []
            for digest, vec in zip(want, vecs):
                v = np.asarray(vec, dtype=np.float32)
                self._vectors[digest] = v
                self._vector_model[digest] = key
                rows.append(self._row(digest, key, v))
            self._append(rows)
            return len(want)

    # ---- ranking -----------------------------------------------------

    def rank(self, query: str, digests: Iterable[str], *,
             semantic: bool = True) -> list[dict[str, Any]]:
        """Fused ranking of ``digests`` (each must be noted) for ``query``.
        Each entry: digest, score, keyword_rank, semantic_rank, similarity,
        exact_name. Exact-name hits lead the keyword list (each at rank 1,
        full weight), the other keyword matches follow in BM25 order at
        ``KEYWORD_WEIGHT``; the semantic list counts at ``SEMANTIC_WEIGHT``."""
        cands = sorted(set(d for d in digests if d in self._tokens))
        kw = bm25_ranking(query, {d: self._tokens[d] for d in cands})
        sem: list[tuple[str, float]] = []
        if semantic:
            self.embed_pending(cands)
            qv = self._embed([query])
            if qv is not None and len(qv):
                q = np.asarray(qv[0], dtype=np.float32)
                with self._lock:
                    have = [d for d in cands if self.has_vector(d)]
                    if have:
                        mat = np.vstack([self._vectors[d] for d in have])
                        sims = mat @ q
                        sem = [(d, round(float(s), 6)) for d, s in zip(have, sims)
                               if float(s) >= SEMANTIC_MIN]
                sem.sort(key=lambda p: (-p[1], p[0]))
        bm25_pos = {d: i for i, (d, _) in enumerate(kw)}
        exact = sorted((d for d in cands
                        if exact_name_hit(query, self._names.get(d, ""))),
                       key=lambda d: (bm25_pos.get(d, len(kw)), d))
        exact_set = set(exact)
        kw_list = exact + [d for d, _ in kw if d not in exact_set]
        kw_rank = {d: 1 if d in exact_set else i
                   for i, d in enumerate(kw_list, start=1)}
        sem_rank = {d: i for i, (d, _) in enumerate(sem, start=1)}
        sim = dict(sem)
        terms: dict[str, list[tuple[float, int]]] = {}
        for d in kw_list:
            w = EXACT_NAME_WEIGHT if d in exact_set else KEYWORD_WEIGHT
            terms.setdefault(d, []).append((w, kw_rank[d]))
        for d, r in sem_rank.items():
            terms.setdefault(d, []).append((SEMANTIC_WEIGHT, r))
        out = []
        for digest, score in fuse(terms):
            out.append({"digest": digest, "score": score,
                        "keyword_rank": kw_rank.get(digest),
                        "semantic_rank": sem_rank.get(digest),
                        "similarity": sim.get(digest),
                        "exact_name": digest in exact_set})
        return out

    # ---- persistence -------------------------------------------------

    @staticmethod
    def _row(digest: str, key: str, vec: np.ndarray) -> str:
        raw = np.asarray(vec, dtype="<f4").tobytes()
        return json.dumps({"digest": digest, "model": key,
                           "v": base64.b64encode(raw).decode("ascii")})

    def _append(self, rows: list[str]) -> None:
        if self.path is None or not rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(rows) + "\n")

    def _load(self) -> None:
        """Load the current model's vectors; rows from any other model are
        dropped (and the file rewritten without them)."""
        if self.path is None or not self.path.exists():
            return
        key = self._model_key()
        stale = 0
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            log.warning("tool search index unreadable: %s", exc)
            return
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row.get("model") != key:
                    stale += 1
                    continue
                vec = np.frombuffer(base64.b64decode(row["v"]), dtype="<f4")
                if vec.shape[0] != minilm.MODEL_DIM:
                    stale += 1
                    continue
                self._vectors[str(row["digest"])] = vec.astype(np.float32)
                self._vector_model[str(row["digest"])] = key
            except (ValueError, KeyError, TypeError):
                stale += 1
        if stale:
            log.info("tool search index: %d vector(s) from another model "
                     "dropped, to be embedded again", stale)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text("".join(
                self._row(d, key, v) + "\n" for d, v in sorted(self._vectors.items())),
                encoding="utf-8")
            os.replace(tmp, self.path)


# ---------------------------------------------------------------------------
# The runtime's index and the manifests it searches
# ---------------------------------------------------------------------------

def index_for(runtime: Any) -> ToolSearchIndex:
    """The runtime's search index (created on first use; tests may pre-set
    ``runtime.tool_search_index``)."""
    index = runtime.__dict__.get("tool_search_index")
    if isinstance(index, ToolSearchIndex):
        return index
    path = None
    try:
        path = Path(runtime._config.data_dir) / "search" / "tool_vectors.jsonl"
    except Exception:  # noqa: BLE001 (no data dir: in memory)
        path = None
    index = ToolSearchIndex(path)
    runtime.__dict__["tool_search_index"] = index
    return index


def _library(runtime: Any) -> Any:
    autonet = getattr(runtime, "autonet", None)
    service = getattr(autonet, "_service", None) if autonet else None
    return getattr(service, "_library", None) if service else None


def network_manifests(runtime: Any) -> dict[str, dict[str, Any]]:
    """Every manifest in the substrate's discovery index (this daemon's
    published tools and every remote manifest it fetched and verified),
    read from the blob store once per digest."""
    lib = _library(runtime)
    if lib is None:
        return {}
    cache: dict[str, Optional[dict[str, Any]]] = runtime.__dict__.setdefault(
        "_tool_search_blob_cache", {})
    try:
        digests = lib.artifact_digests()
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, dict[str, Any]] = {}
    for digest in digests:
        if digest not in cache:
            try:
                payload = lib.blob_json(digest)
            except Exception:  # noqa: BLE001
                payload = None
            cache[digest] = payload if is_tool_manifest(payload) else None
        manifest = cache[digest]
        if manifest is not None:
            out[digest] = manifest
    return out


def candidate_manifests(runtime: Any, caller_id: Optional[str]) -> dict[str, dict[str, Any]]:
    """What a caller may find: every network manifest the daemon holds,
    plus the local tool store's records the caller may see (an agent never
    learns the names of private tools it may not call). The daemon's own
    harness modules (``atn_*``) are not library tools."""
    out = dict(network_manifests(runtime))
    store = getattr(runtime, "tool_store", None)
    if store is not None:
        for record in store.visible_to(caller_id):
            out.setdefault(record.digest, record.manifest)
    return {d: m for d, m in out.items()
            if not str(m.get("name") or "").startswith("atn_")}


# ---------------------------------------------------------------------------
# This household's relation to a tool
# ---------------------------------------------------------------------------

def household_reviews(runtime: Any) -> dict[str, dict[str, Any]]:
    """digest -> this household's newest review of the tool: the reviews
    held for any agent on this daemon (not yet on chain) and those in any
    batch of its batch store that still counts (not forfeited, not rolled
    forward). External sessions share the operator's batch, so this is the
    household's view. Each value: ``axes``, ``harm``, ``state`` ("held" or
    the batch status, "pending" before commit) and ``epoch``. Cached until
    the held reviews or the batches change: every write to them bumps
    ``review_batch.writes()`` (the directories' mtimes stay in the key for
    anything written from outside, but alone they miss changes on exFAT and
    within their timestamp resolution)."""
    from .runtime import review_batch
    try:
        batches = review_batch.store_for(runtime)
        batch_dir = batches._dir
    except Exception:  # noqa: BLE001 (no data dir: no batches)
        batches, batch_dir = None, None
    try:
        held_dir = Path(runtime._config.data_dir) / review_batch.HELD_DIR
    except Exception:  # noqa: BLE001
        held_dir = None

    def mtime(p: Any) -> float:
        try:
            return p.stat().st_mtime if p is not None else 0.0
        except OSError:
            return 0.0

    key = (review_batch.writes(), mtime(batch_dir), mtime(held_dir))
    hit = runtime.__dict__.get("_household_reviews_cache")
    if hit is not None and hit[0] == key:
        return hit[1]
    out: dict[str, dict[str, Any]] = {}
    if batches is not None:
        for batch in batches.all():           # oldest epoch first
            if batch.status in ("forfeited", "rolled"):
                continue
            for r in batch.reviews:
                out[r.digest.lower()] = {
                    "axes": dict(r.axes), "harm": r.harm,
                    "state": batch.status or "pending", "epoch": batch.epoch}
    held = sorted(review_batch.all_held_reviews(runtime),
                  key=lambda r: int((r or {}).get("held_ts") or 0))
    for raw in held:                          # newer than any batch
        digest = str((raw or {}).get("digest") or "").lower().removeprefix("0x")
        if not digest:
            continue
        axes = raw.get("axes") if isinstance(raw.get("axes"), dict) else {}
        out[digest] = {"axes": dict(axes), "harm": raw.get("harm"),
                       "state": "held", "epoch": None}
    runtime.__dict__["_household_reviews_cache"] = (key, out)
    return out


def household_relation(runtime: Any, digest: str,
                       reviews: Optional[dict[str, dict[str, Any]]] = None
                       ) -> dict[str, Any]:
    """``{adopted, authored, reviewed, review}`` for one tool: adopted here
    (by any agent of this daemon), authored here, and this household's own
    review (its scores and harm) when it made one."""
    store = getattr(runtime, "tool_store", None)
    record = store.get(digest) if store is not None else None
    origin = getattr(record, "origin", "") if record is not None else ""
    if reviews is None:
        reviews = household_reviews(runtime)
    review = reviews.get(str(digest).lower())
    return {"adopted": origin == "adopted",
            "authored": record is not None and origin != "adopted",
            "reviewed": review is not None,
            "review": review}


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------

def semantic_state(runtime: Any) -> dict[str, Any]:
    """Whether semantic search runs, and when not, the model's progress."""
    from .search_model import status_for
    ready = minilm.shared() is not None
    if ready:
        return {"ready": True}
    st = status_for(runtime)
    if st.get("state") == "downloading":
        why = (f"the search model is downloading ({st.get('percent', 0):.0f}%, "
               f"{int(st.get('bytes_done') or 0) // 2**20}/"
               f"{int(st.get('bytes_total') or 0) // 2**20} MB)")
    elif st.get("state") == "retrying":
        why = (f"fetching the search model failed ({st.get('error')}); "
               f"retrying in {st.get('retry_in') or 0:.0f}s")
    elif st.get("state") in ("checking", "loading"):
        why = "the search model is loading"
    else:
        why = "the search model is not loaded"
    return {"ready": False, "state": st.get("state"),
            "percent": st.get("percent"), "bytes_done": st.get("bytes_done"),
            "bytes_total": st.get("bytes_total"),
            "note": f"Semantic search is not ready yet: {why}. These are "
                    "keyword matches only; search again later for matches "
                    "by meaning."}


def _base_row(digest: str, manifest: dict[str, Any], hit: dict[str, Any]) -> dict[str, Any]:
    return {
        "digest": digest,
        "name": str(manifest.get("name") or ""),
        "description": str(manifest.get("description") or ""),
        "author": str(manifest.get("author") or ""),
        "trust_class": str(manifest.get("trust_class") or ""),
        "score": hit["score"],
        "match": {"keyword_rank": hit["keyword_rank"],
                  "semantic_rank": hit["semantic_rank"],
                  "similarity": hit["similarity"],
                  "exact_name": hit["exact_name"]},
    }


async def search_tools(runtime: Any, query: str, k: int = 8, *,
                       caller_id: Optional[str] = None) -> dict[str, Any]:
    """The library search behind ``probe_tools`` (agents, the agent API and
    the app's Tools screen). Returns ``{matches, semantic}``: matches in
    relevance order with their metadata; ``semantic.ready`` False (with a
    note and the model's progress) when the results are keyword-only."""
    from .tool_security import annotate_probe_rows

    index = index_for(runtime)
    manifests = candidate_manifests(runtime, caller_id)
    for digest, manifest in manifests.items():
        index.note(digest, manifest)
    semantic = semantic_state(runtime)
    ranked = await asyncio.to_thread(
        index.rank, query, list(manifests), semantic=semantic["ready"])
    rows = [_base_row(h["digest"], manifests[h["digest"]], h)
            for h in ranked[:k]]
    rows = await annotate_probe_rows(runtime, rows)
    reviews = household_reviews(runtime)
    for row in rows:
        row["household"] = household_relation(runtime, row["digest"], reviews)
    return {"matches": rows, "semantic": semantic}


def warm(runtime: Any) -> None:
    """Embed every manifest the daemon holds, on a background thread (the
    model just loaded: manifests learned before it are embedded now, so the
    first semantic query does not pay for them)."""
    def _run() -> None:
        try:
            index = index_for(runtime)
            for digest, manifest in candidate_manifests(runtime, None).items():
                index.note(digest, manifest)
            n = index.embed_pending()
            if n:
                log.info("tool search: %d manifest(s) embedded", n)
        except Exception:  # noqa: BLE001
            log.debug("tool search warm-up failed", exc_info=True)
    threading.Thread(target=_run, name="tool-search-warm", daemon=True).start()
