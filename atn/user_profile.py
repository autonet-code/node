"""User profile persistence.

Two stores live here:

1. The **dossier** — ``data_dir/USER.md``, a markdown document that is the
   fleet-readable record of who the user is (background, skills, goals,
   constraints, values).  Section-granular read/write via the profile tool
   bundle (get_user_profile / update_user_profile).  Hand-editable; there is
   no "onboarding done" state — the dossier is only ever current or stale,
   which readers judge from the inline dates agents are instructed to keep.

2. ``data_dir/profile.json`` — structured plumbing (projects, standards,
   jurisdiction_id).  Goals are tracked as agents in the agent registry —
   creating an agent IS setting a goal.  The onboarding_status field is
   LEGACY-WIRE: atn_web still reads it over WS; nothing in the daemon gates
   on it any more.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import OnboardingStatus, UserProfile

log = logging.getLogger(__name__)


class UserProfileStore:
    """Manages the user profile with JSON persistence.

    Usage:
        store = UserProfileStore(data_dir)
        profile = store.load()
        profile.summary = "..."
        store.save()
    """

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "profile.json"
        data_dir.mkdir(parents=True, exist_ok=True)
        self._profile: UserProfile | None = None

    def load(self) -> UserProfile:
        """Load profile from disk, creating a default if missing."""
        if self._profile is not None:
            return self._profile

        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                self._profile = self._from_dict(raw)
            except Exception:
                log.warning("Failed to load profile from %s, creating default", self._path, exc_info=True)
                self._profile = UserProfile()
        else:
            self._profile = UserProfile()

        return self._profile

    def save(self, profile: UserProfile | None = None) -> None:
        """Persist profile to disk."""
        if profile is not None:
            self._profile = profile
        if self._profile is None:
            return
        self._profile.updated_at = datetime.now(timezone.utc)
        try:
            self._path.write_text(
                json.dumps(self._to_dict(self._profile), indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            log.exception("Failed to save profile to %s", self._path)

    def get_profile(self) -> UserProfile:
        """Get current profile, loading if needed."""
        if self._profile is None:
            self.load()
        return self._profile  # type: ignore[return-value]

    def skip_onboarding(self) -> None:
        """LEGACY-WIRE: mark the legacy onboarding flag completed (atn_web
        'skip' button).  Nothing in the daemon gates on the flag."""
        p = self.get_profile()
        p.onboarding_status = OnboardingStatus.COMPLETED
        self.save()
        log.info("Onboarding skipped")

    # ------------------------------------------------------------------
    # Dossier (USER.md) — section-granular markdown store
    # ------------------------------------------------------------------
    # Sections are H2 blocks ("## Title").  Text before the first H2 (the
    # H1 title line, contact block, etc.) is the preamble and is preserved
    # verbatim.  Lookup is case-insensitive on the H2 title.

    @property
    def dossier_path(self) -> Path:
        return self._path.parent / "USER.md"

    def read_dossier(self) -> str:
        try:
            return self.dossier_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""
        except Exception:
            log.exception("Failed to read dossier at %s", self.dossier_path)
            return ""

    @staticmethod
    def _split_dossier(text: str) -> tuple[str, list[tuple[str, str]]]:
        """Split into (preamble, [(title, body)]).  Body excludes the H2 line."""
        preamble_lines: list[str] = []
        sections: list[tuple[str, list[str]]] = []
        current: list[str] | None = None
        for line in text.splitlines():
            if line.startswith("## "):
                sections.append((line[3:].strip(), []))
                current = sections[-1][1]
            elif current is not None:
                current.append(line)
            else:
                preamble_lines.append(line)
        return (
            "\n".join(preamble_lines),
            [(t, "\n".join(body).strip("\n")) for t, body in sections],
        )

    @staticmethod
    def _join_dossier(preamble: str, sections: list[tuple[str, str]]) -> str:
        parts = [preamble.rstrip("\n")] if preamble.strip() else []
        for title, body in sections:
            parts.append(f"## {title}\n\n{body.strip()}" if body.strip() else f"## {title}")
        return "\n\n".join(parts) + "\n"

    def dossier_sections(self) -> list[str]:
        _, sections = self._split_dossier(self.read_dossier())
        return [t for t, _ in sections]

    def read_dossier_section(self, name: str) -> str | None:
        _, sections = self._split_dossier(self.read_dossier())
        want = name.strip().lower()
        for title, body in sections:
            if title.lower() == want:
                return body
        return None

    def write_dossier_section(self, name: str, content: str = "",
                              mode: str = "replace") -> dict[str, Any]:
        """Write one dossier section.  mode: replace | append | remove.

        A missing section is created (replace/append); remove of a missing
        section is a no-op reported as such.  Returns {section, mode, sections}.
        """
        name = name.strip()
        if not name:
            raise ValueError("Section name is required")
        if mode not in ("replace", "append", "remove"):
            raise ValueError(f"Unknown mode '{mode}' (replace | append | remove)")
        preamble, sections = self._split_dossier(self.read_dossier())
        want = name.lower()
        idx = next((i for i, (t, _) in enumerate(sections) if t.lower() == want), None)

        found = idx is not None
        if mode == "remove":
            if found:
                sections.pop(idx)
        elif idx is None:
            sections.append((name, content))
        elif mode == "append":
            title, body = sections[idx]
            sections[idx] = (title, f"{body}\n\n{content}".strip("\n"))
        else:
            sections[idx] = (sections[idx][0], content)

        self.dossier_path.write_text(
            self._join_dossier(preamble, sections), encoding="utf-8")
        return {
            "section": name,
            "mode": mode,
            "existed": found,
            "sections": [t for t, _ in sections],
        }

    def update_goal(self, goal_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Update a goal by ID.  Returns the updated goal or None if not found."""
        p = self.get_profile()
        for g in p.goals:
            if g.get("id") == goal_id:
                g.update(updates)
                self.save()
                return g
        return None

    def add_goal(self, goal: dict[str, Any]) -> dict[str, Any]:
        """Add a new goal.  Assigns an ID if missing."""
        if "id" not in goal:
            goal["id"] = uuid4().hex[:8]
        if "status" not in goal:
            goal["status"] = "active"
        p = self.get_profile()
        p.goals.append(goal)
        self.save()
        return goal

    def remove_goal(self, goal_id: str) -> bool:
        """Remove a goal by ID.  Returns True if found and removed."""
        p = self.get_profile()
        before = len(p.goals)
        p.goals = [g for g in p.goals if g.get("id") != goal_id]
        if len(p.goals) < before:
            self.save()
            return True
        return False

    def update_project(self, project_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Update a project by ID.  Returns the updated project or None if not found."""
        p = self.get_profile()
        for proj in p.projects:
            if proj.get("id") == project_id:
                proj.update(updates)
                self.save()
                return proj
        return None

    def add_project(self, project: dict[str, Any]) -> dict[str, Any]:
        """Add a new project.  Assigns an ID if missing."""
        if "id" not in project:
            project["id"] = uuid4().hex[:8]
        if "status" not in project:
            project["status"] = "active"
        p = self.get_profile()
        p.projects.append(project)
        self.save()
        return project

    def remove_project(self, project_id: str) -> bool:
        """Remove a project by ID.  Returns True if found and removed."""
        p = self.get_profile()
        before = len(p.projects)
        p.projects = [proj for proj in p.projects if proj.get("id") != project_id]
        if len(p.projects) < before:
            self.save()
            return True
        return False

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a lightweight summary for the snapshot/WS."""
        p = self.get_profile()
        return {
            "onboarding_status": p.onboarding_status.value,
            "summary": p.summary,
            "project_count": len(p.projects),
            "jurisdiction_id": p.jurisdiction_id,
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dict(p: UserProfile) -> dict[str, Any]:
        return {
            "id": p.id,
            "onboarding_status": p.onboarding_status.value,
            "summary": p.summary,
            "standards": p.standards,
            "goals": p.goals,
            "projects": p.projects,
            "strengths": p.strengths,
            "weaknesses": p.weaknesses,
            "jurisdiction_id": p.jurisdiction_id,
            "created_at": p.created_at.isoformat(),
            "updated_at": p.updated_at.isoformat(),
        }

    @staticmethod
    def _from_dict(d: dict[str, Any]) -> UserProfile:
        status_str = d.get("onboarding_status", "not_started")
        try:
            status = OnboardingStatus(status_str)
        except ValueError:
            status = OnboardingStatus.NOT_STARTED

        created = d.get("created_at")
        updated = d.get("updated_at")

        return UserProfile(
            id=d.get("id", "local"),
            onboarding_status=status,
            summary=d.get("summary", ""),
            standards=d.get("standards", []),
            goals=d.get("goals", []),
            projects=d.get("projects", []),
            strengths=d.get("strengths", []),
            weaknesses=d.get("weaknesses", []),
            jurisdiction_id=d.get("jurisdiction_id"),
            created_at=datetime.fromisoformat(created) if created else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(updated) if updated else datetime.now(timezone.utc),
        )
