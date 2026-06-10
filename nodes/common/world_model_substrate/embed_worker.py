"""Out-of-process embedding worker.

Why this exists: importing sentence_transformers/torch from a daemon
worker thread has wedged on Windows (loader-lock interaction during
extension-DLL initialization, poisoning the process-wide import
machinery; see usefulness_coords). Importing the same stack on the
MAIN thread of a fresh process is reliable — so the daemon spawns
this script and never imports torch itself.

Protocol (stdio, one JSON object per line):

  worker -> parent   {"ready": true, "backend": "..."}    once, after init
  parent -> worker   {"text": "..."}
  worker -> parent   {"coords": [...]}  |  {"error": "..."}
  parent -> worker   (EOF)                                 worker exits

Run directly by file path — deliberately NOT through the nodes
package (whose __init__ imports web3 and friends):

  python embed_worker.py <dim> [model_name] [backend]

backend "hashing" exists for protocol tests: it serves the
dependency-free HashingEmbedder instead of sentence-transformers.

usefulness_coords.py is loaded from this script's own directory via
importlib spec loading, again to avoid package-level imports.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_usefulness_coords():
    path = Path(__file__).resolve().parent / "usefulness_coords.py"
    spec = importlib.util.spec_from_file_location("uc_standalone", path)
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec: dataclasses resolves string annotations via
    # sys.modules[cls.__module__] and the file uses future annotations.
    sys.modules["uc_standalone"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    dim = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    model_name = sys.argv[2] if len(sys.argv) > 2 else (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    backend = sys.argv[3] if len(sys.argv) > 3 else "st"

    uc = _load_usefulness_coords()
    if backend == "hashing":
        embedder = uc.HashingEmbedder(dim=dim)
    else:
        embedder = uc.SentenceTransformersEmbedder(dim=dim, model_name=model_name)
        embedder("warmup")   # force model load before declaring ready

    sys.stdout.write(json.dumps({"ready": True, "backend": backend}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            coords = embedder(str(req.get("text", "")))
            out = {"coords": list(coords)}
        except Exception as e:  # keep serving; parent decides what to do
            out = {"error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
