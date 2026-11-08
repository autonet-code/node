"""The substrate charter as data (docs/multi_jurisdiction.md decision 5).

A jurisdiction's DAO Registry holds its charter text under
``jurisdiction.<id>.charter``: compact JSON ``{"axes":[{"id","thesis"},...]}``.
Each DAO defines its own axes; ``correctness`` and ``simplicity`` are
mandatory (discovery ranks on them). The axis order in the text is the
coordinate order: axis ``i`` is charter coordinate ``i``.

One jurisdiction runs per process, so one charter is active per process.
``set_active_charter`` installs it at join, before any world is built; the
default is the v1 six-axis charter. A charter change takes effect at the
next epoch through a daemon restart (the close driver detects it pinned to
the previous anchor), never by swapping the charter under a live world.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

MANDATORY_AXES = ("correctness", "simplicity")
# Same rules werule and the on-chain ToolBond enforce.
MAX_AXES = 16
_AXIS_ID_RE = re.compile(r"^[a-z0-9_]+$")


@dataclass(frozen=True)
class Charter:
    """An ordered set of charter axes, each ``(id, thesis)``."""

    axes: Tuple[Tuple[str, str], ...]
    _text: str = field(default="", compare=False, repr=False)

    def __post_init__(self) -> None:
        axes = tuple((str(i), str(t)) for i, t in self.axes)
        object.__setattr__(self, "axes", axes)
        if not self._text:
            object.__setattr__(self, "_text", _serialize(axes))

    # -- shape ---------------------------------------------------------------

    @property
    def n_dims(self) -> int:
        return len(self.axes)

    @property
    def ids(self) -> Tuple[str, ...]:
        return tuple(i for i, _ in self.axes)

    @property
    def axis_index(self) -> Dict[str, int]:
        return {aid: i for i, (aid, _) in enumerate(self.axes)}

    def entries(self) -> List[Dict[str, Any]]:
        """Dicts shaped like the legacy ``adapter.CHARTER`` entries."""
        return [
            {"id": aid, "thesis": thesis, "axis_index": i, "veto_floor": None}
            for i, (aid, thesis) in enumerate(self.axes)
        ]

    # -- identity ------------------------------------------------------------

    @property
    def text(self) -> str:
        """The registry bytes (compact JSON) as a str."""
        return self._text

    @property
    def hash(self) -> str:
        """``0x`` + sha256 of ``text``: the Registry's charter hash."""
        return "0x" + hashlib.sha256(self._text.encode("utf-8")).hexdigest()

    # -- parsing -------------------------------------------------------------

    @classmethod
    def from_text(cls, text: str) -> "Charter":
        """Parse and validate registry charter text. Raises ValueError."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("charter text is empty")
        if not text.isascii():
            raise ValueError("charter text must be pure ASCII")
        try:
            parsed = json.loads(text)
        except Exception as e:
            raise ValueError(f"charter is not valid JSON: {e}") from None
        if not isinstance(parsed, dict) or not isinstance(parsed.get("axes"), list):
            raise ValueError('charter must be a JSON object {"axes": [...]}')
        axes: List[Tuple[str, str]] = []
        for a in parsed["axes"]:
            if (not isinstance(a, dict) or not isinstance(a.get("id"), str)
                    or not isinstance(a.get("thesis"), str) or not a["id"]):
                raise ValueError(
                    'every charter axis must be {"id": <non-empty str>, '
                    '"thesis": <str>}')
            axes.append((a["id"], a["thesis"]))
        if not axes:
            raise ValueError("charter has no axes")
        if len(axes) > MAX_AXES:
            raise ValueError(
                f"charter has {len(axes)} axes; at most {MAX_AXES} allowed")
        bad = [i for i, _ in axes if not _AXIS_ID_RE.fullmatch(i)]
        if bad:
            raise ValueError(
                "charter axis ids must match ^[a-z0-9_]+$: " + ", ".join(bad))
        ids = [i for i, _ in axes]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"charter has duplicate axis ids: {', '.join(dupes)}")
        missing = [m for m in MANDATORY_AXES if m not in ids]
        if missing:
            raise ValueError(
                "charter lacks the mandatory axes: " + ", ".join(missing))
        # ToolBond counts axes on chain by counting "id": in the text, which
        # is only reliable for the compact form; refuse anything else so a
        # hand-written proposal can't make the contract and daemons disagree.
        if json.dumps(parsed, separators=(",", ":"), ensure_ascii=False) != text:
            raise ValueError("charter is not in compact JSON form")
        return cls(axes=tuple(axes), _text=text)


def _serialize(axes: Tuple[Tuple[str, str], ...]) -> str:
    return json.dumps(
        {"axes": [{"id": i, "thesis": t} for i, t in axes]},
        separators=(",", ":"), ensure_ascii=False,
    )


# The v1 charter: 4 alignment axes plus the 2 mandatory usefulness axes.
DEFAULT_CHARTER = Charter(axes=(
    ("life_precious", "Life is precious and should be preserved."),
    ("self_preservation", "The system should preserve its own continuity."),
    ("promotion_of_intelligence", "Intelligence in any form should be promoted."),
    ("evolution", "Forward advancement of capability is desirable."),
    ("correctness", "Work should achieve what it claims and avoid bugs."),
    ("simplicity", "Solutions should be minimal and direct, not over-engineered."),
))

_lock = threading.Lock()
_active: Charter = DEFAULT_CHARTER


def active_charter() -> Charter:
    """The charter this process runs (DEFAULT_CHARTER until set)."""
    return _active


def set_active_charter(charter: Charter) -> None:
    """Install the process-wide charter. Call before building any world."""
    global _active
    if not isinstance(charter, Charter):
        raise TypeError("set_active_charter expects a Charter")
    with _lock:
        _active = charter
