#!/usr/bin/env python3
"""Phase 10 task batteries — ground-truth case data + the real execution
contract runner.

Prereg: docs/phase10_prereg.md (committed 6da0264). Each of the 36 tasks
has a battery of >= 10 cases: 10 generated happy-path cases plus 1-2
hand-built edge cases per defect kind applicable to the task. Expected
values are computed by build_tools.py from the HONEST implementation
(single source of truth); this module holds only the case *inputs* and
the runner.

Execution contract: the real ToolStore pinned path
(atn/tool_store.py::_call_pinned) runs a code blob as
``[sys.executable, script]`` with JSON args on stdin and JSON result on
stdout. ``run_tool`` below reproduces exactly that contract (bare
python, no guard wrapper — these are authored tools, not adopted ones).

No LLM calls anywhere (prereg: "The machine is measured, not a model's
opinion of it").
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

TOOL_TIMEOUT_S = 20.0

# Words used to build happy-path inputs. ASCII, short (keeps happy
# strings under the boundary-defect length), disjoint from the
# retrieval-side instance wordlists in build_retrieval.py.
WORDS = [
    "amber", "basil", "cedar", "delta", "ember", "fjord", "grove",
    "heron", "ivory", "jasper", "kelp", "lotus", "maple", "nectar",
    "olive", "pearl", "quartz", "raven", "sage", "topaz",
]


# ---------------------------------------------------------------------------
# Happy-path generators — one per task, deterministic given the rng.
# Constraints (so happy cases can never trip an implanted defect):
# strings are ASCII and <= 40 chars, lists have <= 8 items, numbers are
# positive, primary inputs are non-empty.
# ---------------------------------------------------------------------------


def _words(rng, lo=3, hi=5) -> List[str]:
    return [rng.choice(WORDS) for _ in range(rng.randint(lo, hi))]


def _sentence(rng, lo=3, hi=5) -> str:
    s = " ".join(_words(rng, lo, hi))
    return s[:40]


def _nums(rng, lo=3, hi=8, top=99) -> List[int]:
    return [rng.randint(1, top) for _ in range(rng.randint(lo, hi))]


def _records(rng, lo=3, hi=6) -> List[Dict[str, Any]]:
    teams = ["red", "blue", "gold"]
    return [
        {"team": rng.choice(teams), "score": rng.randint(1, 50)}
        for _ in range(rng.randint(lo, hi))
    ]


HAPPY_GENERATORS = {
    # -- strings ----------------------------------------------------------
    "reverse_words": lambda rng: {"text": _sentence(rng)},
    "title_case": lambda rng: {"text": _sentence(rng)},
    "count_vowels": lambda rng: {"text": _sentence(rng)},
    "snake_to_camel": lambda rng: {"text": "_".join(_words(rng, 2, 4))},
    "squeeze_spaces": lambda rng: {
        "text": (" " * rng.randint(1, 3)).join(_words(rng, 2, 4))},
    "rot13_encode": lambda rng: {"text": _sentence(rng, 2, 4)},
    "is_palindrome": lambda rng: (
        {"text": (lambda w: w + w[::-1])(rng.choice(WORDS))}
        if rng.random() < 0.5 else {"text": _sentence(rng, 2, 3)}),
    "dedupe_words": lambda rng: {
        "text": " ".join(_words(rng, 3, 5) + [rng.choice(WORDS)] * 2)[:40]},
    "longest_word": lambda rng: {"text": _sentence(rng)},
    # -- parsing ----------------------------------------------------------
    "parse_kv": lambda rng: {"text": ";".join(
        f"{k}={rng.randint(1, 99)}" for k in _words(rng, 2, 4))},
    "csv_fields": lambda rng: {"text": ",".join(_words(rng, 3, 5))},
    "extract_ints": lambda rng: {"text": " ".join(
        w if i % 2 else str(rng.randint(1, 99))
        for i, w in enumerate(_words(rng, 3, 5)))},
    "query_params": lambda rng: {"text": "&".join(
        f"{k}={rng.randint(1, 99)}" for k in _words(rng, 2, 4))},
    "hex_to_int": lambda rng: {
        "text": format(rng.randint(1, 65535), "x")},
    "parse_bool_flag": lambda rng: {
        "text": rng.choice(["yes", "no", "true", "false", "on", "off",
                            "1", "0", "y", "n"])},
    "count_lines": lambda rng: {"text": "\n".join(_words(rng, 2, 5))},
    "extract_emails": lambda rng: {"text": " ".join([
        rng.choice(WORDS),
        f"{rng.choice(WORDS)}@{rng.choice(WORDS)}.com",
        rng.choice(WORDS)])[:40]},
    "first_number": lambda rng: {"text": " ".join([
        rng.choice(WORDS), str(rng.randint(1, 99)),
        rng.choice(WORDS), str(rng.randint(1, 99))])},
    # -- numbers ----------------------------------------------------------
    "sum_list": lambda rng: {"values": _nums(rng)},
    "mean_value": lambda rng: {"values": _nums(rng)},
    "range_span": lambda rng: {"values": _nums(rng)},
    "clamp_value": lambda rng: {
        "value": rng.randint(1, 99), "lo": rng.randint(1, 20),
        "hi": rng.randint(60, 99)},
    "add_days": lambda rng: {
        "date": f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        "days": rng.randint(1, 45)},
    "weekday_name": lambda rng: {
        "date": f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"},
    "median_value": lambda rng: {"values": _nums(rng)},
    "digit_sum": lambda rng: {"value": rng.randint(1, 99999)},
    "count_positive": lambda rng: {"values": _nums(rng)},
    # -- munging ----------------------------------------------------------
    "dedupe_values": lambda rng: {
        "values": _nums(rng, 3, 6) + [7, 7]},
    "sort_records": lambda rng: {"records": _records(rng), "key": "score"},
    "flatten_lists": lambda rng: {
        "values": [_nums(rng, 1, 3, top=20) for _ in range(rng.randint(2, 4))]},
    "count_by_field": lambda rng: {"records": _records(rng), "key": "team"},
    "filter_at_least": lambda rng: {
        "values": _nums(rng), "threshold": rng.randint(10, 60)},
    "top_values": lambda rng: {"values": _nums(rng), "n": rng.randint(1, 3)},
    "pair_up": lambda rng: (lambda ws: {
        "keys": ws, "values": list(range(1, len(ws) + 1))})(_words(rng, 2, 5)),
    "reverse_items": lambda rng: {"values": _nums(rng)},
    "total_by_field": lambda rng: {
        "records": _records(rng), "key": "team", "value_key": "score"},
}


# ---------------------------------------------------------------------------
# Edge cases — hand-built per (task, defect kind). Each case is an input
# on which the named defect kind CHANGES the tool's output (asserted at
# build time by build_tools.py against the honest implementation).
# Tags: "edge_empty", "edge_unicode", "edge_negative", "edge_boundary".
# The full battery for a task includes every edge case listed here,
# whether or not that task's tool instance is defective — the battery is
# the task's SPEC, identical for defective and correct tools (guard #1:
# nothing in battery structure reveals which tool carries a defect).
# ---------------------------------------------------------------------------

_LONG_SENT = "amber basil cedar delta ember fjord golf o"          # len 42
_LONG_PAL = "abcdefghijklmnopqrstuvwxyzyxwvutsrqponmlkjihgfedcba"  # len 51

EDGE_CASES: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    # -- strings ----------------------------------------------------------
    "reverse_words": {
        "empty": [{"text": ""}],
        "unicode": [{"text": "café νερό water"}],
        "boundary": [{"text": _LONG_SENT}],
    },
    "title_case": {
        "empty": [{"text": ""}],
        "unicode": [{"text": "élan vital"}],
        "boundary": [{"text": _LONG_SENT}],
    },
    "count_vowels": {
        "empty": [{"text": ""}],
        "unicode": [{"text": "café olé"}],
        "boundary": [{"text": _LONG_SENT}],
    },
    "snake_to_camel": {
        "empty": [{"text": ""}],
        "unicode": [{"text": "café_bar_menu"}],
        "boundary": [{"text": "alpha_bravo_charlie_delta_echo_foxtrot_gh"}],
    },
    "squeeze_spaces": {
        "empty": [{"text": ""}],
        "unicode": [{"text": "a  é  b"}],
        "boundary": [{"text": "amber   basil   cedar   delta   ember  qq"}],
    },
    "rot13_encode": {
        "empty": [{"text": ""}],
        "unicode": [{"text": "café rot"}],
        "boundary": [{"text": _LONG_SENT}],
    },
    "is_palindrome": {
        "empty": [{"text": ""}],
        "unicode": [{"text": "aé"}],
        "boundary": [{"text": _LONG_PAL}],
    },
    "dedupe_words": {
        "empty": [{"text": ""}],
        "unicode": [{"text": "café caf café"}],
        "boundary": [{"text": _LONG_SENT}],
    },
    "longest_word": {
        "empty": [{"text": ""}],
        "unicode": [{"text": "résumé note"}],
        "boundary": [{"text": "amber basil cedar delta extraordinary abc"}],
    },
    # -- parsing ----------------------------------------------------------
    "parse_kv": {
        "empty": [{"text": ""}],
        "unicode": [{"text": "café=1;b=2"}],
        "boundary": [{"text": "aa=1;bb=2;cc=3;dd=4;ee=5;ff=6;gg=7;hh=890"}],
    },
    "csv_fields": {
        "empty": [{"text": ""}],
        "unicode": [{"text": "a,café,b"}],
        "boundary": [{"text": "amber,basil,cedar,delta,ember,fjord,heron"}],
    },
    "extract_ints": {
        "empty": [{"text": ""}],
        "negative": [{"text": "temp -5 rose to 7"},
                     {"text": "delta -12 and -3"}],
        "boundary": [{"text": "amber 11 basil 22 cedar 33 delta 44 ee 250"}],
    },
    "query_params": {
        "empty": [{"text": ""}],
        "unicode": [{"text": "café=1&b=2"}],
        "boundary": [{"text": "aa=1&bb=2&cc=3&dd=4&ee=5&ff=6&gg=7&hh=890"}],
    },
    "hex_to_int": {
        "empty": [{"text": ""}],
        "negative": [{"text": "-ff"}, {"text": "-1a"}],
        "boundary": [{"text": "1" + "0" * 41}],
    },
    "parse_bool_flag": {
        "empty": [{"text": ""}],
    },
    "count_lines": {
        "empty": [{"text": ""}],
        "unicode": [{"text": "é\n\nnote"}],
        "boundary": [{"text": "amber basil cedar delta ember fjord\nab\nx"}],
    },
    "extract_emails": {
        "empty": [{"text": ""}],
        "unicode": [{"text": "reach josé@mail.com today"}],
        "boundary": [{"text": "write to amber.basil@cedar.com and hi@x.io"}],
    },
    "first_number": {
        "empty": [{"text": ""}],
        "negative": [{"text": "loss of -2.5 percent"},
                     {"text": "shift -14 units"}],
        "boundary": [{"text": "amber basil cedar delta ember fjord as 250"}],
    },
    # -- numbers ----------------------------------------------------------
    "sum_list": {
        "empty": [{"values": []}],
        "negative": [{"values": [3, -4, 5]}, {"values": [-1, -2, 9]}],
        "boundary": [{"values": list(range(1, 42))}],
    },
    "mean_value": {
        "empty": [{"values": []}],
        "negative": [{"values": [6, -2]}, {"values": [-8, 4, 1]}],
        "boundary": [{"values": list(range(1, 41)) + [999]}],
    },
    "range_span": {
        "empty": [{"values": []}],
        "negative": [{"values": [-5, 3]}, {"values": [-9, -1, 2]}],
        "boundary": [{"values": list(range(1, 41)) + [999]}],
    },
    "clamp_value": {
        "negative": [{"value": -7, "lo": -5, "hi": 5},
                     {"value": -30, "lo": -10, "hi": 10}],
    },
    "add_days": {
        "empty": [{"date": "", "days": 3}],
        "negative": [{"date": "2024-03-10", "days": -3},
                     {"date": "2024-01-05", "days": -10}],
    },
    "weekday_name": {
        "empty": [{"date": ""}],
    },
    "median_value": {
        "empty": [{"values": []}],
        "negative": [{"values": [-9, 1, 5]}, {"values": [2, -2]}],
        "boundary": [{"values": list(range(1, 42))}],
    },
    "digit_sum": {
        "negative": [{"value": -45}, {"value": -908}],
    },
    "count_positive": {
        "empty": [{"values": []}],
        "negative": [{"values": [-1, 2, -3]}, {"values": [-4, -5]}],
        "boundary": [{"values": [0] * 40 + [7]}],
    },
    # -- munging ----------------------------------------------------------
    "dedupe_values": {
        "empty": [{"values": []}],
        "negative": [{"values": [2, -2, 3]}, {"values": [-1, 1]}],
        "boundary": [{"values": list(range(1, 41)) + [1, 99]}],
    },
    "sort_records": {
        "empty": [{"records": [], "key": "score"}],
        "boundary": [{"records": [{"team": "t", "score": 50 + i}
                                  for i in range(41)] + [
                                      {"team": "z", "score": 1}],
                      "key": "score"}],
    },
    "flatten_lists": {
        "empty": [{"values": []}],
        "boundary": [{"values": [[i] for i in range(41)] + [[99]]}],
    },
    "count_by_field": {
        "empty": [{"records": [], "key": "team"}],
        "boundary": [{"records": [{"team": "a"}] * 41 + [{"team": "b"}],
                      "key": "team"}],
    },
    "filter_at_least": {
        "empty": [{"values": [], "threshold": 3}],
        "negative": [{"values": [-5, 2, 8], "threshold": 0},
                     {"values": [-3, -1, 4], "threshold": 2}],
        "boundary": [{"values": [1] * 41 + [50], "threshold": 40}],
    },
    "top_values": {
        "empty": [{"values": [], "n": 2}],
        "negative": [{"values": [-9, 5, 1], "n": 2},
                     {"values": [-20, 3], "n": 1}],
        "boundary": [{"values": [1] * 41 + [50], "n": 1}],
    },
    "pair_up": {
        "empty": [{"keys": [], "values": []}],
        "boundary": [{"keys": [f"k{i}" for i in range(42)],
                      "values": list(range(42))}],
    },
    "reverse_items": {
        "empty": [{"values": []}],
        "negative": [{"values": [-1, 2, 3]}, {"values": [4, -5]}],
        "boundary": [{"values": list(range(1, 43))}],
    },
    "total_by_field": {
        "empty": [{"records": [], "key": "team", "value_key": "score"}],
        "boundary": [{"records": [{"team": "a", "score": 1}] * 41 + [
                          {"team": "b", "score": 9}],
                      "key": "team", "value_key": "score"}],
    },
}

N_HAPPY = 10


def build_battery(task_name: str, rng) -> List[Dict[str, Any]]:
    """Assemble the case list for one task: N_HAPPY generated happy
    cases (tag "happy") + every hand-built edge case (tag "edge_<kind>").
    Expected values are attached later by build_tools.py from the honest
    implementation. Deterministic given the rng state.
    """
    gen = HAPPY_GENERATORS[task_name]
    cases: List[Dict[str, Any]] = [
        {"args": gen(rng), "tag": "happy"} for _ in range(N_HAPPY)
    ]
    for kind in sorted(EDGE_CASES.get(task_name, {})):
        for args in EDGE_CASES[task_name][kind]:
            cases.append({"args": args, "tag": f"edge_{kind}"})
    return cases


# ---------------------------------------------------------------------------
# Runner — the real pinned execution contract (stdin JSON -> stdout JSON)
# ---------------------------------------------------------------------------


def run_tool(code_path: Path, args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a pinned tool blob exactly like ToolStore._call_pinned:
    ``[sys.executable, script]``, JSON args on stdin, JSON on stdout.

    Returns {"ok": bool, "output": parsed-or-None, "error": str}.
    """
    try:
        proc = subprocess.run(
            [sys.executable, str(code_path)],
            input=json.dumps(args).encode("utf-8"),
            capture_output=True,
            timeout=TOOL_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": None,
                "error": f"timeout after {TOOL_TIMEOUT_S}s"}
    if proc.returncode != 0:
        return {"ok": False, "output": None,
                "error": proc.stderr.decode("utf-8", errors="replace")[:500]}
    out_text = proc.stdout.decode("utf-8", errors="replace").strip()
    try:
        return {"ok": True, "output": json.loads(out_text), "error": ""}
    except json.JSONDecodeError:
        return {"ok": False, "output": None,
                "error": f"non-JSON stdout: {out_text[:200]}"}


def _norm(value: Any) -> Any:
    """JSON round-trip normalization so in-process expected values
    compare exactly against subprocess-parsed outputs."""
    return json.loads(json.dumps(value))


def run_battery(code_path: Path, battery: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run every battery case through the real execution contract.

    Each case must carry ``args``, ``expected``, ``tag``. Returns
    {"passed", "total", "pass_rate", "cases": [{tag, passed}]}.
    A case passes iff the subprocess succeeds AND its parsed output
    equals the expected value (JSON equality).
    """
    results = []
    passed = 0
    for case in battery:
        r = run_tool(code_path, case["args"])
        ok = bool(r["ok"]) and r["output"] == _norm(case["expected"])
        passed += 1 if ok else 0
        results.append({"tag": case["tag"], "passed": ok})
    total = len(battery)
    return {
        "passed": passed,
        "total": total,
        "pass_rate": (passed / total) if total else 0.0,
        "cases": results,
    }


def replay_invocation(code_path: Path, invocation: Dict[str, Any]) -> bool:
    """Replay a CON's attached invocation against the pinned code.

    ``invocation`` = {"args": ..., "expected": ...}. Returns True iff
    the failure REPRODUCES: the tool errors out or its output differs
    from the invocation's expected value. Deterministic (the corpus
    tools are pure functions of stdin).
    """
    r = run_tool(code_path, invocation["args"])
    if not r["ok"]:
        return True
    return r["output"] != _norm(invocation["expected"])


def battery_for(corpus_tool: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The battery list persisted for a tool in corpus.json."""
    return corpus_tool["battery"]


if __name__ == "__main__":
    print(f"{len(HAPPY_GENERATORS)} happy generators, "
          f"{len(EDGE_CASES)} edge tables")
    missing = set(HAPPY_GENERATORS) ^ set(EDGE_CASES)
    if missing:
        print(f"MISMATCH: {sorted(missing)}")
        raise SystemExit(1)
