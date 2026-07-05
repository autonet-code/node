#!/usr/bin/env python3
"""Phase 10 corpus generator — deterministic, seeded, executable ground truth.

Prereg: docs/phase10_prereg.md (committed 6da0264). Master seed 1010 is
the ONLY sanctioned seed (guard #4); this module refuses any other unless
an amendment commit precedes it.

Produces, into experiments/phase10/tools/ and corpus.json:

  - ~40 pinned tool code blobs (stdin-JSON -> stdout-JSON, the real
    ToolStore._call_pinned contract), across 4 task families:
    strings, parsing, numbers, munging.
  - Half carry an IMPLANTED runtime-only DEFECT (edge-case failure that
    passes the happy-path battery subset). Guard #1: the manifest
    DESCRIPTION is generated from the HONEST behavior template regardless
    of whether the instance is defective — a description reader cannot
    tell. build.py asserts the string distance between a defective tool's
    manifest text and its correct sibling's is zero for the same task.
  - Adversarial salting (H2/H3): SEO manifests (query-vocabulary
    descriptions, unrelated behavior) and wash tools (junk behavior, no
    real coverage — the sybil-attestation adversary is added at debate/
    receipt build time, this marks the class).
  - The battery expected values, computed by RUNNING the honest
    implementation of each task through the real subprocess contract
    (batteries.run_tool). Ground-truth defectiveness = does the tool's
    pass_rate fall below 1.0 on its own battery.

Everything is a pure function of the master seed: same seed -> byte-
identical corpus.json and identical tool blobs on disk (guard #3 relies
on this; test_harness asserts it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from batteries import (
    HAPPY_GENERATORS,
    build_battery,
    run_tool,
)

MASTER_SEED = 1010
HERE = Path(__file__).resolve().parent
TOOLS_DIR = HERE / "tools"
CORPUS_PATH = HERE / "corpus.json"

# Task families (prereg: >= 4). Names must match batteries.HAPPY_GENERATORS.
FAMILIES: Dict[str, List[str]] = {
    "strings": [
        "reverse_words", "title_case", "count_vowels", "snake_to_camel",
        "squeeze_spaces", "rot13_encode", "is_palindrome", "dedupe_words",
        "longest_word",
    ],
    "parsing": [
        "parse_kv", "csv_fields", "extract_ints", "query_params",
        "hex_to_int", "parse_bool_flag", "count_lines", "extract_emails",
        "first_number",
    ],
    "numbers": [
        "sum_list", "mean_value", "range_span", "clamp_value", "add_days",
        "weekday_name", "median_value", "digit_sum", "count_positive",
    ],
    "munging": [
        "dedupe_values", "sort_records", "flatten_lists", "count_by_field",
        "filter_at_least", "top_values", "pair_up", "reverse_items",
        "total_by_field",
    ],
}

ALL_TASKS: List[str] = [t for tasks in FAMILIES.values() for t in tasks]

# One-line manifest DESCRIPTION per task, written from the HONEST behavior
# (guard #1: identical for the defective and correct instance of a task).
DESCRIPTIONS: Dict[str, str] = {
    "reverse_words": "Reverse the order of whitespace-separated words in the input text.",
    "title_case": "Capitalize the first letter of each word in the input text.",
    "count_vowels": "Count the number of vowels (aeiou) in the input text.",
    "snake_to_camel": "Convert a snake_case identifier to camelCase.",
    "squeeze_spaces": "Collapse runs of whitespace in the text to single spaces.",
    "rot13_encode": "Apply the ROT13 letter substitution cipher to the text.",
    "is_palindrome": "Report whether the text reads the same forwards and backwards.",
    "dedupe_words": "Remove duplicate words, preserving first-seen order.",
    "longest_word": "Return the longest whitespace-separated word in the text.",
    "parse_kv": "Parse a semicolon-separated key=value string into a mapping.",
    "csv_fields": "Split a comma-separated line into a list of fields.",
    "extract_ints": "Extract every integer appearing in the text, in order.",
    "query_params": "Parse an ampersand-separated URL query string into a mapping.",
    "hex_to_int": "Convert a hexadecimal string to its integer value.",
    "parse_bool_flag": "Interpret a textual flag as a boolean.",
    "count_lines": "Count the number of lines in the text.",
    "extract_emails": "Extract every email address appearing in the text.",
    "first_number": "Return the first integer appearing in the text.",
    "sum_list": "Return the sum of a list of numbers.",
    "mean_value": "Return the arithmetic mean of a list of numbers.",
    "range_span": "Return the difference between the max and min of the list.",
    "clamp_value": "Clamp a value into the inclusive [lo, hi] range.",
    "add_days": "Add a number of days to an ISO date, returning the new date.",
    "weekday_name": "Return the weekday name for an ISO date.",
    "median_value": "Return the median of a list of numbers.",
    "digit_sum": "Return the sum of the decimal digits of an integer.",
    "count_positive": "Count the strictly positive numbers in the list.",
    "dedupe_values": "Remove duplicate values, preserving first-seen order.",
    "sort_records": "Sort records ascending by a numeric key field.",
    "flatten_lists": "Flatten a list of lists into a single list.",
    "count_by_field": "Count records grouped by a field's value.",
    "filter_at_least": "Keep values greater than or equal to a threshold.",
    "top_values": "Return the n largest values, descending.",
    "pair_up": "Zip a list of keys with a list of values into a mapping.",
    "reverse_items": "Reverse the order of items in a list.",
    "total_by_field": "Sum a value field within each group of a key field.",
}

# The QUERY vocabulary a genuine user would type for each task — the H2
# retrieval side embeds these. Kept here so the SEO adversary can borrow
# a DIFFERENT task's query vocabulary (claim one thing, do another).
TASK_QUERY_TERMS: Dict[str, str] = {
    "reverse_words": "reverse the words in a sentence backwards order",
    "title_case": "capitalize each word title case heading",
    "count_vowels": "count vowels letters in a string",
    "snake_to_camel": "convert snake case to camel case identifier",
    "squeeze_spaces": "collapse multiple spaces whitespace into one",
    "rot13_encode": "rot13 cipher encode letters substitution",
    "is_palindrome": "check palindrome reads same backwards",
    "dedupe_words": "remove duplicate repeated words unique",
    "longest_word": "find the longest word in text",
    "parse_kv": "parse key value pairs semicolon mapping",
    "csv_fields": "split comma separated csv fields line",
    "extract_ints": "extract all integers numbers from text",
    "query_params": "parse url query string parameters ampersand",
    "hex_to_int": "convert hexadecimal hex string to integer number",
    "parse_bool_flag": "parse boolean flag yes no true false",
    "count_lines": "count number of lines in text newline",
    "extract_emails": "extract email addresses from text",
    "first_number": "get the first integer number in text",
    "sum_list": "sum add a list of numbers total",
    "mean_value": "average arithmetic mean of numbers",
    "range_span": "range span difference max minus min",
    "clamp_value": "clamp bound a value between lo hi limits",
    "add_days": "add days to an iso date calendar",
    "weekday_name": "weekday day name for a date",
    "median_value": "median middle value of numbers",
    "digit_sum": "sum of the digits of an integer",
    "count_positive": "count positive numbers greater than zero",
    "dedupe_values": "remove duplicate values unique list",
    "sort_records": "sort records by numeric key field ascending",
    "flatten_lists": "flatten nested list of lists into one",
    "count_by_field": "count records grouped by field value",
    "filter_at_least": "filter values at least threshold minimum",
    "top_values": "top n largest values descending",
    "pair_up": "zip pair keys with values into mapping",
    "reverse_items": "reverse the order of items in a list",
    "total_by_field": "sum total value field grouped by key",
}


# ---------------------------------------------------------------------------
# Honest + defective code templates. Each task has one HONEST body and, when
# a defect kind applies, a DEFECTIVE body that diverges only on the edge
# case (the happy-path battery subset still passes — that is the whole
# point: a description reader and a happy-path tester both miss it).
#
# Every tool is a self-contained script: read JSON from stdin, print JSON
# to stdout. Kept tiny so the ~40-tool battery runs in minutes.
# ---------------------------------------------------------------------------

_HEADER = "import sys, json\nd = json.load(sys.stdin)\n"


def _emit(body: str) -> str:
    return _HEADER + body


# honest[task] -> code ; defect[task] -> (kind, code). Absent from defect
# = task has no defective variant (it will only ever be a correct tool).
HONEST: Dict[str, str] = {
    "reverse_words": _emit(
        "print(json.dumps(' '.join(d['text'].split()[::-1])))"),
    "title_case": _emit(
        "print(json.dumps(' '.join(w[:1].upper()+w[1:] for w in d['text'].split(' '))))"),
    "count_vowels": _emit(
        "print(json.dumps(sum(c in 'aeiouAEIOU' for c in d['text'])))"),
    "snake_to_camel": _emit(
        "p=d['text'].split('_'); print(json.dumps(p[0]+''.join(w[:1].upper()+w[1:] for w in p[1:])))"),
    "squeeze_spaces": _emit(
        "import re; print(json.dumps(re.sub(r'\\s+',' ',d['text']).strip()))"),
    "rot13_encode": _emit(
        "import codecs; print(json.dumps(codecs.encode(d['text'],'rot_13')))"),
    "is_palindrome": _emit(
        "s=d['text']; print(json.dumps(s==s[::-1]))"),
    "dedupe_words": _emit(
        "seen=[]; [seen.append(w) for w in d['text'].split() if w not in seen]; print(json.dumps(' '.join(seen)))"),
    "longest_word": _emit(
        "ws=d['text'].split(); print(json.dumps(max(ws,key=len) if ws else ''))"),
    "parse_kv": _emit(
        "out={}; [out.__setitem__(*p.split('=',1)) for p in d['text'].split(';') if '=' in p]; print(json.dumps(out))"),
    "csv_fields": _emit(
        "print(json.dumps(d['text'].split(',') if d['text'] else []))"),
    "extract_ints": _emit(
        "import re; print(json.dumps([int(x) for x in re.findall(r'-?\\d+',d['text'])]))"),
    "query_params": _emit(
        "out={}; [out.__setitem__(*p.split('=',1)) for p in d['text'].split('&') if '=' in p]; print(json.dumps(out))"),
    "hex_to_int": _emit(
        "print(json.dumps(int(d['text'],16) if d['text'] else 0))"),
    "parse_bool_flag": _emit(
        "print(json.dumps(d['text'].strip().lower() in ('yes','true','on','1','y')))"),
    "count_lines": _emit(
        "print(json.dumps(len(d['text'].split(chr(10))) if d['text'] else 0))"),
    "extract_emails": _emit(
        "import re; print(json.dumps(re.findall(r'[\\w.]+@[\\w.]+',d['text'])))"),
    "first_number": _emit(
        "import re; m=re.search(r'-?\\d+',d['text']); print(json.dumps(int(m.group()) if m else None))"),
    "sum_list": _emit(
        "print(json.dumps(sum(d['values'])))"),
    "mean_value": _emit(
        "v=d['values']; print(json.dumps(sum(v)/len(v) if v else 0))"),
    "range_span": _emit(
        "v=d['values']; print(json.dumps(max(v)-min(v) if v else 0))"),
    "clamp_value": _emit(
        "print(json.dumps(max(d['lo'],min(d['hi'],d['value']))))"),
    "add_days": _emit(
        "import datetime as dt; base=dt.date.fromisoformat(d['date']); print(json.dumps((base+dt.timedelta(days=d['days'])).isoformat()))"),
    "weekday_name": _emit(
        "import datetime as dt; print(json.dumps(dt.date.fromisoformat(d['date']).strftime('%A')))"),
    "median_value": _emit(
        "v=sorted(d['values']); n=len(v); print(json.dumps(0 if n==0 else (v[n//2] if n%2 else (v[n//2-1]+v[n//2])/2)))"),
    "digit_sum": _emit(
        "print(json.dumps(sum(int(c) for c in str(abs(d['value'])))))"),
    "count_positive": _emit(
        "print(json.dumps(sum(1 for x in d['values'] if x>0)))"),
    "dedupe_values": _emit(
        "seen=[]; [seen.append(x) for x in d['values'] if x not in seen]; print(json.dumps(seen))"),
    "sort_records": _emit(
        "print(json.dumps(sorted(d['records'],key=lambda r:r[d['key']])))"),
    "flatten_lists": _emit(
        "print(json.dumps([x for sub in d['values'] for x in sub]))"),
    "count_by_field": _emit(
        "out={}; [out.__setitem__(r[d['key']],out.get(r[d['key']],0)+1) for r in d['records']]; print(json.dumps(out))"),
    "filter_at_least": _emit(
        "print(json.dumps([x for x in d['values'] if x>=d['threshold']]))"),
    "top_values": _emit(
        "print(json.dumps(sorted(d['values'],reverse=True)[:d['n']]))"),
    "pair_up": _emit(
        "print(json.dumps(dict(zip(d['keys'],d['values']))))"),
    "reverse_items": _emit(
        "print(json.dumps(d['values'][::-1]))"),
    "total_by_field": _emit(
        "out={}; [out.__setitem__(r[d['key']],out.get(r[d['key']],0)+r[d['value_key']]) for r in d['records']]; print(json.dumps(out))"),
}

# Defective variants: (defect_kind, code). The body diverges from HONEST
# only on the named edge kind; happy-path cases still pass.
DEFECT: Dict[str, Tuple[str, str]] = {
    # empty-input crashes / wrong answers -------------------------------
    "reverse_words": ("boundary", _emit(
        "ws=d['text'].split(); print(json.dumps(' '.join(ws[:6][::-1])))")),  # drops words past 6
    "count_vowels": ("boundary", _emit(
        "print(json.dumps(sum(c in 'aeiouAEIOU' for c in d['text'][:40])))")),  # ignores chars past 40 (boundary case is len 42)
    "title_case": ("boundary", _emit(
        "print(json.dumps(' '.join(w[:1].upper()+w[1:] for w in d['text'].split(' ')[:6])))")),  # drops words past 6 (boundary case has 8)
    "squeeze_spaces": ("unicode", _emit(
        "print(json.dumps(' '.join(d['text'].split(' ')).strip()))")),  # only collapses ' ', not tabs/newlines... (here all-space, differs on runs? honest uses \s+; this splits single space -> keeps empty tokens? use .split() honest)
    "is_palindrome": ("boundary", _emit(
        "s=d['text']; print(json.dumps(s[:20]==s[:20][::-1]))")),  # only checks first 20 chars
    "dedupe_words": ("boundary", _emit(
        "print(json.dumps(' '.join(dict.fromkeys(d['text'].split()[:6]))))")),  # truncates
    "longest_word": ("boundary", _emit(
        "ws=[w for w in d['text'].split() if len(w)<=10]; print(json.dumps(max(ws,key=len) if ws else ''))")),  # ignores long words
    "parse_kv": ("boundary", _emit(
        "out={}; [out.__setitem__(*p.split('=',1)) for p in d['text'].split(';')[:6] if '=' in p]; print(json.dumps(out))")),  # drops pairs past 6
    "extract_ints": ("negative", _emit(
        "import re; print(json.dumps([int(x) for x in re.findall(r'\\d+',d['text'])]))")),  # loses minus sign
    "hex_to_int": ("negative", _emit(
        "t=d['text']; print(json.dumps(int(t.lstrip('-'),16) if t else 0))")),  # drops the sign: -ff -> 255 not -255
    "first_number": ("negative", _emit(
        "import re; m=re.search(r'\\d+',d['text']); print(json.dumps(int(m.group()) if m else None))")),  # loses sign
    "csv_fields": ("boundary", _emit(
        "print(json.dumps(d['text'].split(',')[:6] if d['text'] else []))")),
    "count_lines": ("boundary", _emit(
        "n=d['text'].count(chr(10)); print(json.dumps(n if d['text'] else 0))")),  # off-by-one: counts separators, not lines
    "extract_emails": ("boundary", _emit(
        "import re; print(json.dumps(re.findall(r'[\\w.]+@[\\w.]+',d['text'])[:1]))")),  # only first email
    "sum_list": ("negative", _emit(
        "print(json.dumps(sum(x for x in d['values'] if x>=0)))")),  # ignores negatives
    "mean_value": ("negative", _emit(
        "v=[x for x in d['values'] if x>=0]; print(json.dumps(sum(v)/len(v) if v else 0))")),
    "range_span": ("negative", _emit(
        "v=[x for x in d['values'] if x>=0]; print(json.dumps(max(v)-min(v) if v else 0))")),  # ignores negatives: span wrong when min is negative
    "clamp_value": ("negative", _emit(
        "print(json.dumps(max(0,min(d['hi'],d['value'])) if d['lo']<0 else max(d['lo'],min(d['hi'],d['value']))))")),  # floors negative lo at 0
    "add_days": ("negative", _emit(
        "import datetime as dt; base=dt.date.fromisoformat(d['date']); dd=max(0,d['days']); print(json.dumps((base+dt.timedelta(days=dd)).isoformat()))")),  # ignores negative days
    "median_value": ("negative", _emit(
        "v=sorted(x for x in d['values'] if x>=0); n=len(v); print(json.dumps(0 if n==0 else (v[n//2] if n%2 else (v[n//2-1]+v[n//2])/2)))")),
    "digit_sum": ("negative", _emit(
        "print(json.dumps(sum(int(c) for c in str(d['value']) if c.isdigit()) if d['value']>=0 else -sum(int(c) for c in str(-d['value']))))")),  # negates the sum for negative input
    "count_positive": ("negative", _emit(
        "print(json.dumps(sum(1 for x in d['values'] if x>=0)))")),  # counts zero as positive
    "dedupe_values": ("boundary", _emit(
        "print(json.dumps(list(dict.fromkeys(d['values']))[:20]))")),
    "flatten_lists": ("boundary", _emit(
        "print(json.dumps([x for sub in d['values'][:20] for x in sub]))")),
    "filter_at_least": ("boundary", _emit(
        "print(json.dumps([x for x in d['values'] if x>d['threshold']]))")),  # strict > instead of >=
    "top_values": ("boundary", _emit(
        "print(json.dumps(sorted(d['values'][:40],reverse=True)[:d['n']]))")),  # only considers first 40 values (boundary case has 42, max at the end)
    "reverse_items": ("boundary", _emit(
        "print(json.dumps(d['values'][:20][::-1]))")),
    "count_by_field": ("boundary", _emit(
        "recs=d['records'][:40]; out={}; [out.__setitem__(r[d['key']],out.get(r[d['key']],0)+1) for r in recs]; print(json.dumps(out))")),
    "sort_records": ("boundary", _emit(
        "recs=d['records'][:40]; print(json.dumps(sorted(recs,key=lambda r:r[d['key']])))")),
    "total_by_field": ("boundary", _emit(
        "recs=d['records'][:40]; out={}; [out.__setitem__(r[d['key']],out.get(r[d['key']],0)+r[d['value_key']]) for r in recs]; print(json.dumps(out))")),
    "pair_up": ("boundary", _emit(
        "ks=d['keys'][:40]; print(json.dumps(dict(zip(ks,d['values']))))")),
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _battery_with_expected(task: str, rng: random.Random) -> List[Dict[str, Any]]:
    """Battery cases + expected values from the HONEST implementation,
    computed via the REAL subprocess contract (single source of truth)."""
    cases = build_battery(task, rng)
    honest_path = TOOLS_DIR / f"_honest_{task}.py"
    # write_bytes (not write_text): Windows write_text rewrites \n -> \r\n,
    # which would desync the on-disk file from _sha(code) and change the
    # subprocess behavior. The real ToolStore writes the code blob as bytes.
    honest_path.write_bytes(HONEST[task].encode("utf-8"))
    out: List[Dict[str, Any]] = []
    for case in cases:
        r = run_tool(honest_path, case["args"])
        # The honest tool defines ground truth. If the honest tool itself
        # errors on a case (a bad edge input), that case is dropped —
        # ground truth must be well-defined.
        if not r["ok"]:
            continue
        out.append({"args": case["args"], "expected": r["output"],
                    "tag": case["tag"]})
    honest_path.unlink()
    return out


def build_corpus(seed: int) -> Dict[str, Any]:
    if seed != MASTER_SEED:
        raise SystemExit(
            f"guard #4 (seed shopping): master seed is {MASTER_SEED}; "
            f"refusing seed {seed}. To change it, land an amendment commit "
            "on docs/phase10_prereg.md FIRST.")
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    tools: List[Dict[str, Any]] = []

    # Class assignment is deterministic and balanced (guard: half defective
    # among the honest-behavior tools). We build, per task:
    #   - one CORRECT tool (honest code),
    #   - and for tasks with a defect variant, one DEFECTIVE tool sharing
    #     the SAME manifest description (guard #1).
    # SEO + wash adversaries are a fixed subset drawn deterministically.
    seo_tasks = set(rng.sample(ALL_TASKS, 6))     # claim vocab != behavior
    wash_tasks = set(rng.sample(sorted(set(ALL_TASKS) - seo_tasks), 6))

    for task in ALL_TASKS:
        family = next(f for f, ts in FAMILIES.items() if task in ts)
        battery = _battery_with_expected(task, rng)

        # ---- correct instance -------------------------------------------
        correct_code = HONEST[task]
        tools.append(_make_tool(
            task, family, "correct", correct_code,
            description=DESCRIPTIONS[task],
            query_terms=TASK_QUERY_TERMS[task],
            battery=battery, seo=False, wash=False))

        # ---- defective instance (same description; guard #1) ------------
        if task in DEFECT:
            defect_kind, defect_code = DEFECT[task]
            tools.append(_make_tool(
                task, family, "defective", defect_code,
                description=DESCRIPTIONS[task],       # HONEST template text
                query_terms=TASK_QUERY_TERMS[task],
                battery=battery, seo=False, wash=False,
                defect_kind=defect_kind))

    # ---- SEO adversaries: correct-behaving tools whose DESCRIPTION and
    # query vocabulary are borrowed from a DIFFERENT task (claim one thing,
    # do another). They behave correctly on their real task's battery, but
    # advertise into the wrong region — the retrieval anti-SEO target.
    for task in sorted(seo_tasks):
        victim = _rot_task(task)
        # SEO tool: real behavior = `victim` task, advertised text = `task`.
        # A task-unique no-op comment keeps the code (and thus the digest)
        # distinct even if two SEO tools happen to share a victim task —
        # identity is the code_digest, and colliding digests would pool
        # their registrations/receipts into one node (a corpus bug).
        battery = _battery_with_expected(victim, rng)
        seo_code = HONEST[victim] + f"# seo:{task}->{victim}\n"
        tools.append(_make_tool(
            victim, next(f for f, ts in FAMILIES.items() if victim in ts),
            "seo", seo_code,
            description=DESCRIPTIONS[task],            # advertises `task`
            query_terms=TASK_QUERY_TERMS[task],       # matches `task` queries
            battery=battery, seo=True, wash=False,
            seo_target_task=task, real_task=victim))

    # ---- wash adversaries: junk tools (a trivial passthrough that fails
    # most of a real task's battery) marketed as a real task. No real
    # coverage; their mint threat is sybil attestation (added in
    # build_debates). Ground-truth quality = low battery pass rate. The
    # junk body carries a task-unique comment so each wash tool has a
    # DISTINCT code_digest (else all wash tools collapse to one node and
    # pool their sybil receipts — a corpus bug the assertion below guards).
    for task in sorted(wash_tasks):
        battery = _battery_with_expected(task, rng)
        junk = _emit(
            f"# wash:{task}\nprint(json.dumps(d.get('text', d.get('values', ''))))")
        tools.append(_make_tool(
            task, next(f for f, ts in FAMILIES.items() if task in ts),
            "wash", junk,
            description=DESCRIPTIONS[task],
            query_terms=TASK_QUERY_TERMS[task],
            battery=battery, seo=False, wash=True))

    # Digest-uniqueness invariant: every tool must have a distinct
    # code_digest (content addressing == identity; a collision would pool
    # separate tools' registrations, receipts, and standing into one node).
    digests = [t["code_digest"] for t in tools]
    if len(digests) != len(set(digests)):
        from collections import Counter
        dupes = {d: n for d, n in Counter(digests).items() if n > 1}
        raise SystemExit(f"corpus digest collision (would pool nodes): {dupes}")

    corpus = {
        "master_seed": seed,
        "n_tools": len(tools),
        "families": {f: len(ts) for f, ts in FAMILIES.items()},
        "tools": tools,
    }
    CORPUS_PATH.write_text(
        json.dumps(corpus, indent=2, sort_keys=True), encoding="utf-8")
    return corpus


def _rot_task(task: str) -> str:
    """Deterministic 'other task' for the SEO victim — the next task in
    ALL_TASKS order, wrapping around. Guarantees behavior != advertised."""
    i = ALL_TASKS.index(task)
    return ALL_TASKS[(i + 7) % len(ALL_TASKS)]


def _make_tool(
    task: str, family: str, trust_kind: str, code: str, *,
    description: str, query_terms: str, battery: List[Dict[str, Any]],
    seo: bool, wash: bool, defect_kind: str = "",
    seo_target_task: str = "", real_task: str = "",
) -> Dict[str, Any]:
    """Materialize a tool blob to disk and return its corpus record.

    Runs the tool's OWN battery through the real subprocess contract to
    measure its ground-truth pass_rate — the defectiveness label is a
    MEASURED quantity, not an assertion. A defect that (by seed accident)
    happened not to fail any battery case would show pass_rate 1.0 and be
    correctly labelled non-defective.
    """
    code_digest = _sha(code)
    code_path = TOOLS_DIR / f"{code_digest}.py"
    # Bytes, not text: content addressing requires the on-disk file to hash
    # to code_digest. Windows write_text would inject \r\n and break that.
    code_path.write_bytes(code.encode("utf-8"))

    # Measure ground truth by running this tool's battery.
    passed = 0
    from batteries import _norm
    for case in battery:
        r = run_tool(code_path, case["args"])
        if r["ok"] and r["output"] == _norm(case["expected"]):
            passed += 1
    total = len(battery)
    pass_rate = passed / total if total else 0.0
    defective = pass_rate < 1.0

    # A synthetic per-tool author id (0x-shaped, so the owner map / damper
    # treat it as a distinct consensus identity). Deterministic in digest.
    author = "0x" + _sha(f"author:{task}:{trust_kind}:{code_digest}")[:40]

    # Manifest embedding text is name+description+schema prop names
    # (manifest_embedding_text). We synthesize a manifest so the retrieval
    # side embeds exactly what production would.
    name = f"{task}_{trust_kind}"
    input_schema = _schema_for(task)

    return {
        "task": task,
        "family": family,
        "trust_kind": trust_kind,           # correct|defective|seo|wash
        "code_digest": code_digest,
        "author": author,
        "name": name,
        "description": description,
        "query_terms": query_terms,
        "input_schema": input_schema,
        "battery": battery,
        "pass_rate": pass_rate,
        "defective": defective,
        "defect_kind": defect_kind,
        "seo": seo,
        "wash": wash,
        "seo_target_task": seo_target_task,
        "real_task": real_task,
    }


def _schema_for(task: str) -> Dict[str, Any]:
    """A minimal JSON-schema for the task's primary input (drives the
    manifest embedding text's property-name component)."""
    sample = HAPPY_GENERATORS[task](random.Random(0))
    props = {k: {"type": "string"} for k in sample}
    return {"type": "object", "properties": props}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=MASTER_SEED)
    args = ap.parse_args()
    corpus = build_corpus(args.seed)
    n_def = sum(1 for t in corpus["tools"] if t["defective"])
    print(f"built {corpus['n_tools']} tools "
          f"({n_def} defective by measured pass_rate) -> {CORPUS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
