"""Tools as codebases (atn/tool_tree.py, docs/tool_codebases.md): snapshot,
tree digest, registration by path, running the verified snapshot."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from atn import tool_tree
from atn.agent_tools import execute_tool

from tests.atn.test_tool_store import _family, _make_runtime

SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}}


def _codebase(root: Path, *, greeting: str = "hi") -> Path:
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "words.py").write_text(f"GREETING = {greeting!r}\n")
    (root / "main.py").write_text(
        "import sys, json\n"
        "from pkg.words import GREETING\n"
        "args = json.load(sys.stdin)\n"
        "print(json.dumps({'said': GREETING + ' ' + args.get('x', '')}))\n")
    (root / "notes.log").write_text("local junk")
    (root / ".git").mkdir(exist_ok=True)
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main")
    (root / tool_tree.MANIFEST_FILE).write_text(json.dumps({
        "name": "greet", "description": "Greets x.", "input_schema": SCHEMA,
        "entrypoint": "main.py", "exclude": ["*.log"],
        "source": {"repo": "https://github.com/example/greet", "commit": "abc123"},
    }))
    return root


def test_snapshot_is_content_addressed(tmp_path):
    root = _codebase(tmp_path / "a")
    meta, digest, archive = tool_tree.snapshot(root)
    files = tool_tree.unpack(archive)
    assert sorted(files) == [tool_tree.MANIFEST_FILE, "main.py",
                             "pkg/__init__.py", "pkg/words.py"]
    assert tool_tree.tree_digest_of(files) == digest
    # Same content elsewhere: same digest. Different content: different.
    other = _codebase(tmp_path / "b")
    assert tool_tree.snapshot(other)[1] == digest
    (other / "pkg" / "words.py").write_text("GREETING = 'yo'\n")
    assert tool_tree.snapshot(other)[1] != digest


def test_unpack_refuses_escapes(tmp_path):
    import io, zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../evil.py", "x")
    with pytest.raises(tool_tree.TreeError):
        tool_tree.unpack(buf.getvalue())


def test_bad_manifests_are_refused(tmp_path):
    root = tmp_path / "c"
    root.mkdir()
    with pytest.raises(tool_tree.TreeError):
        tool_tree.snapshot(root)
    (root / tool_tree.MANIFEST_FILE).write_text(json.dumps({
        "name": "x", "description": "d", "input_schema": SCHEMA,
        "entrypoint": "missing.py"}))
    with pytest.raises(tool_tree.TreeError, match="entrypoint"):
        tool_tree.snapshot(root)


@pytest.mark.asyncio
async def test_register_by_path_and_run_the_snapshot(tmp_path):
    rt = _make_runtime(tmp_path)
    await _family(rt)
    root = _codebase(tmp_path / "work" / "greet")
    res = await execute_tool("register_tool", {"path": str(root)}, rt,
                             caller_id="child")
    assert "error" not in res, res
    snap = res["snapshot"]
    assert snap["files"] == 4 and snap["sent_bytes"] > 0
    assert "exclude" in snap["note"]
    record = rt.tool_store.get(res["digest"])
    m = record.manifest
    assert m["code_format"] == "tree" and m["entrypoint"] == "main.py"
    assert m["source"] == {"repo": "https://github.com/example/greet",
                           "commit": "abc123"}
    assert record.source_path == str(root.resolve())

    out = await rt.tool_store.call(record, {"x": "there"}, caller_id="child")
    assert out["result"] == {"said": "hi there"}, out

    # Someone edits the extracted copy: it is re-verified and restored.
    cached = tmp_path / "data" / "tools" / "code_cache" / "trees" / m["tree_digest"]
    (cached / "pkg" / "words.py").write_text("GREETING = 'pwned'\n")
    (cached / "planted.py").write_text("x = 1\n")
    out = await rt.tool_store.call(record, {"x": "there"}, caller_id="child")
    assert out["result"] == {"said": "hi there"}
    assert not (cached / "planted.py").exists()

    # Editing the source and registering again makes a new version.
    (root / "pkg" / "words.py").write_text("GREETING = 'hello'\n")
    res2 = await execute_tool("register_tool",
                              {"path": str(root), "version_of": res["digest"]},
                              rt, caller_id="child")
    assert res2["digest"] != res["digest"]
    rec2 = rt.tool_store.get(res2["digest"])
    out = await rt.tool_store.call(rec2, {"x": "you"}, caller_id="child")
    assert out["result"] == {"said": "hello you"}


@pytest.mark.asyncio
async def test_single_script_tools_get_their_own_verified_dir(tmp_path):
    from tests.atn.test_tool_store import _register_echo
    rt = _make_runtime(tmp_path)
    await _family(rt)
    res = await _register_echo(rt)
    record = rt.tool_store.get(res["digest"])
    out = await rt.tool_store.call(record, {"x": "a"}, caller_id="child")
    assert out["result"] == {"echo": "a"}
    code = record.manifest["code_digest"]
    script = tmp_path / "data" / "tools" / "code_cache" / "scripts" / code / "tool.py"
    script.write_text("print('{\"echo\": \"tampered\"}')\n")
    out = await rt.tool_store.call(record, {"x": "a"}, caller_id="child")
    assert out["result"] == {"echo": "a"}


@pytest.mark.asyncio
async def test_codebase_cannot_be_the_daemon_data(tmp_path):
    rt = _make_runtime(tmp_path)
    await _family(rt)
    data = tmp_path / "data"
    (data / tool_tree.MANIFEST_FILE).write_text(json.dumps({
        "name": "steal", "description": "d", "input_schema": SCHEMA,
        "entrypoint": "x.py"}))
    res = await execute_tool("register_tool", {"path": str(data)}, rt,
                             caller_id="child")
    assert "daemon's data" in res["error"]


@pytest.mark.asyncio
async def test_a_codebase_tool_is_adopted_and_runs_strict(tmp_path):
    from tests.atn.test_tool_store import TestAdoption, _register_agent
    author = _make_runtime(tmp_path / "author_daemon")
    await _register_agent(author, "foreign-author")
    root = _codebase(tmp_path / "work" / "greet")
    res = await execute_tool("register_tool", {"path": str(root)}, author,
                             caller_id="foreign-author")
    assert "error" not in res, res
    rt = await TestAdoption()._adopter(tmp_path, author.tool_store._blob_store())
    out = await execute_tool("adopt_tool", {"digest": res["digest"]}, rt,
                             caller_id="worker")
    assert out.get("origin") == "adopted", out
    record = rt.tool_store.get(res["digest"])
    result = await rt.tool_store.call(record, {"x": "net"}, caller_id="worker")
    assert result["result"] == {"said": "hi net"}, result
    assert result["containment"]["tier"] == "strict"


@pytest.mark.asyncio
async def test_vet_tool_shows_a_codebase_as_files(tmp_path):
    rt = _make_runtime(tmp_path)
    await _family(rt)
    root = _codebase(tmp_path / "work" / "greet")
    res = await execute_tool("register_tool", {"path": str(root)}, rt,
                             caller_id="child")
    out = await execute_tool("vet_tool", {"digest": res["digest"]}, rt,
                             caller_id="parent")
    assert out["files"]["pkg/words.py"].strip() == "GREETING = 'hi'", out
    assert "main.py" in out["files"] and out["code"] is None


@pytest.mark.asyncio
async def test_adopt_with_path_checks_the_code_out(tmp_path):
    from tests.atn.test_tool_store import TestAdoption, _register_agent
    author = _make_runtime(tmp_path / "author_daemon")
    await _register_agent(author, "foreign-author")
    root = _codebase(tmp_path / "work" / "greet")
    res = await execute_tool("register_tool", {"path": str(root)}, author,
                             caller_id="foreign-author")
    rt = await TestAdoption()._adopter(tmp_path, author.tool_store._blob_store())
    dest = tmp_path / "mine" / "greet"
    out = await execute_tool("adopt_tool", {"digest": res["digest"], "path": str(dest)},
                             rt, caller_id="worker")
    assert out["checkout"]["path"] == str(dest.resolve()), out
    assert (dest / "pkg" / "words.py").read_text().strip() == "GREETING = 'hi'"
    assert not (dest / "notes.log").exists()      # it was never in the snapshot
    assert "attest" in out["note"]
    # The checkout registers again as is (same content, same tree digest).
    assert tool_tree.snapshot(dest)[1] == rt.tool_store.get(res["digest"]).manifest["tree_digest"]

    # Already installed: a second checkout elsewhere; a non-empty folder is refused.
    again = await execute_tool("adopt_tool", {"digest": res["digest"],
                                              "path": str(tmp_path / "copy2")},
                               rt, caller_id="worker")
    assert again["checkout"]["files"] == out["checkout"]["files"]
    busy = await execute_tool("adopt_tool", {"digest": res["digest"], "path": str(dest)},
                              rt, caller_id="worker")
    assert "not an empty directory" in busy["checkout_error"]


@pytest.mark.asyncio
async def test_a_single_script_checks_out_as_a_codebase(tmp_path):
    from tests.atn.test_tool_store import TestAdoption
    author, digest, blobs = await TestAdoption()._publish_on_author_daemon(tmp_path)
    rt = await TestAdoption()._adopter(tmp_path, blobs)
    dest = tmp_path / "echo"
    out = await execute_tool("adopt_tool", {"digest": digest, "path": str(dest)},
                             rt, caller_id="worker")
    assert out["checkout"]["entrypoint"] == "main.py", out
    meta = json.loads((dest / tool_tree.MANIFEST_FILE).read_text())
    assert meta["entrypoint"] == "main.py" and meta["name"] == "net_tool"
    assert tool_tree.snapshot(dest)[0]["name"] == "net_tool"


@pytest.mark.asyncio
async def test_publish_reauthors_a_tool_registered_before_the_agent(tmp_path, monkeypatch):
    """Registered while the agent had no on-chain identity: once it has
    one, the store re-authors the same tool under it (a new digest)."""
    rt = _make_runtime(tmp_path)
    await _family(rt)
    root = _codebase(tmp_path / "work" / "greet")
    res = await execute_tool("register_tool", {"path": str(root)}, rt,
                             caller_id="child")
    store = rt.tool_store
    addr = rt.get_agent("child").identity.address
    monkeypatch.setattr(store, "_consensus_identity", lambda author: addr)
    out = store.restamp_author(res["digest"])
    new = store.get(out["digest"])
    assert out["digest"] != res["digest"] and new.manifest["author"] == addr
    assert new.manifest["tree_digest"] == tool_tree.snapshot(root)[1]
    assert new.source_path == str(root.resolve())
    assert out["replaced"] == res["digest"] and store.get(res["digest"]) is None


@pytest.mark.asyncio
async def test_search_leaves_out_the_harness_modules(tmp_path):
    rt = _make_runtime(tmp_path)
    await _family(rt)
    await execute_tool("register_tool", {"path": str(_codebase(tmp_path / "g"))},
                       rt, caller_id="child")
    out = await execute_tool("probe_tools", {"query": "greets shell profile", "k": 10}, rt)
    names = [m["name"] for m in out["matches"]]
    assert "greet" in names and not any(n.startswith("atn_") for n in names)


@pytest.mark.asyncio
async def test_register_on_chain_explains_the_owner_step(tmp_path):
    rt = _make_runtime(tmp_path)
    await _family(rt)
    out = await execute_tool("register_on_chain", {}, rt, caller_id="grandparent")
    assert "error" in out and "'agent_id'" not in out["error"]

def test_snapshot_follows_gitignore(tmp_path):
    root = _codebase(tmp_path / "a")
    (root / ".gitignore").write_text(".env\nbuild/\n*.tmp\n!keep.tmp\n")
    (root / ".env").write_text("SECRET=1")
    (root / "build").mkdir()
    (root / "build" / "out.py").write_text("x")
    (root / "a.tmp").write_text("x")
    (root / "keep.tmp").write_text("x")
    (root / "pkg" / ".gitignore").write_text("local.py\n")
    (root / "pkg" / "local.py").write_text("x")
    files = tool_tree.list_files(root, ["*.log"])
    assert ".env" not in files and "a.tmp" not in files
    assert not any(f.startswith("build/") for f in files)
    assert "pkg/local.py" not in files
    assert "keep.tmp" in files and ".gitignore" in files


def test_snapshot_follows_gitignore_above_the_tool(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".gitignore").write_text("*.env\ntools/greet/scratch/\n")
    root = repo / "tools" / "greet"
    root.mkdir(parents=True)
    (root / "main.py").write_text("print('{}')")
    (root / "prod.env").write_text("SECRET=1")
    (root / "scratch").mkdir()
    (root / "scratch" / "x.py").write_text("x")
    (root / tool_tree.MANIFEST_FILE).write_text(json.dumps({
        "name": "g", "description": "d", "input_schema": SCHEMA,
        "entrypoint": "main.py"}))
    assert tool_tree.list_files(root) == [tool_tree.MANIFEST_FILE, "main.py"]
