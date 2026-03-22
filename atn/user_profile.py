"""User profile persistence.

Stores the user's onboarding state, goals, projects, strengths, weaknesses,
and summary.  JSON file at ``data_dir/profile.json``.

The profile is the foundation for the goal-oriented planning loop: the
orchestrator reads it to understand what the user wants, and updates it as
goals evolve.
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

    def needs_onboarding(self) -> bool:
        """Check if user hasn't completed onboarding."""
        return self.get_profile().onboarding_status != OnboardingStatus.COMPLETED

    def complete_onboarding(self, results: dict[str, Any]) -> None:
        """Mark onboarding as completed and populate profile from results."""
        p = self.get_profile()
        p.onboarding_status = OnboardingStatus.COMPLETED
        p.summary = results.get("summary", "")
        p.standards = results.get("standards", [])
        p.strengths = results.get("strengths", [])
        p.weaknesses = results.get("weaknesses", [])

        # Goals — assign IDs if missing
        for g in results.get("goals", []):
            if "id" not in g:
                g["id"] = uuid4().hex[:8]
            if "status" not in g:
                g["status"] = "active"
        p.goals = results.get("goals", [])

        # Projects — assign IDs if missing
        for proj in results.get("projects", []):
            if "id" not in proj:
                proj["id"] = uuid4().hex[:8]
            if "status" not in proj:
                proj["status"] = "active"
        p.projects = results.get("projects", [])

        self.save()
        log.info("Onboarding completed: %d goals, %d projects", len(p.goals), len(p.projects))

    def skip_onboarding(self) -> None:
        """Mark onboarding as completed with empty data."""
        p = self.get_profile()
        p.onboarding_status = OnboardingStatus.COMPLETED
        self.save()
        log.info("Onboarding skipped")

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
            "goal_count": len(p.goals),
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
