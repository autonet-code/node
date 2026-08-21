"""MCP server for Google Calendar and Google Tasks.

Exposes read/write access to Google Calendar events and Google Tasks
as MCP tools.  Agents discover these tools via MCP and use
them on behalf of the user.

OAuth tokens are passed as environment variables by the ConnectorManager:
  - access_token:  current bearer token
  - refresh_token: long-lived refresh token

Token refresh is handled transparently when API calls return 401.

Runs as a subprocess launched by ConnectorManager.  Communicates with
the runtime via MCP stdio protocol.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

# All logging to stderr — stdout is reserved for MCP protocol.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gcal.mcp")

# -- Configuration -----------------------------------------------------------

GCAL_BASE = "https://www.googleapis.com/calendar/v3"
GTASKS_BASE = "https://tasks.googleapis.com/tasks/v1"

CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"

# Mutable token state — refreshed automatically on 401.
_access_token: str = ""
_refresh_token: str = ""

# -- MCP Server --------------------------------------------------------------

mcp = FastMCP("Google Calendar")


# -- Token helpers ------------------------------------------------------------

def _load_tokens() -> None:
    """Load tokens from environment variables."""
    global _access_token, _refresh_token
    _access_token = os.environ.get("access_token", "")
    _refresh_token = os.environ.get("refresh_token", "")
    if not _access_token:
        log.warning("no access_token in environment")


def _do_refresh() -> bool:
    """Refresh the access token using the refresh token.  Returns True on success."""
    global _access_token
    if not _refresh_token:
        log.error("no refresh_token available — cannot refresh")
        return False

    data = urllib.parse.urlencode({
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": _refresh_token,
        "grant_type": "refresh_token",
    }).encode()
    req = urllib.request.Request(TOKEN_ENDPOINT, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
        _access_token = body.get("access_token", "")
        log.info("access token refreshed successfully")
        return True
    except Exception:
        log.exception("token refresh failed")
        return False


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_access_token}"}


async def _gcal_request(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any] | list[Any]:
    """Make an authenticated request to a Google API.  Retries once on 401."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.request(
            method, url, headers=_auth_headers(), params=params, json=json_body,
        )

        if resp.status_code == 401 and _do_refresh():
            resp = await client.request(
                method, url, headers=_auth_headers(), params=params, json=json_body,
            )

        if resp.status_code >= 400:
            return {"error": resp.text, "status_code": resp.status_code}

        if resp.status_code == 204:
            return {"ok": True}

        return resp.json()


# -- Calendar tools -----------------------------------------------------------


@mcp.tool()
async def list_calendars() -> str:
    """List all calendars the user has access to.

    Returns calendar IDs, names, and access roles.  Use the calendar ID
    in other tools to target a specific calendar (default is "primary").
    """
    result = await _gcal_request("GET", f"{GCAL_BASE}/users/me/calendarList")
    return json.dumps(result, indent=2)


@mcp.tool()
async def list_events(
    time_min: str = "",
    time_max: str = "",
    max_results: int = 25,
    calendar_id: str = "primary",
    query: str = "",
) -> str:
    """List upcoming calendar events.

    Args:
        time_min: Start of time range in ISO 8601 format (e.g. "2025-01-15T00:00:00Z").
                  Defaults to now.
        time_max: End of time range in ISO 8601 format.
                  Defaults to 7 days from now.
        max_results: Maximum number of events to return (1-100, default 25).
        calendar_id: Calendar ID (default "primary").
        query: Free-text search query to filter events.
    """
    now = datetime.now(timezone.utc)
    if not time_min:
        time_min = now.isoformat()
    if not time_max:
        time_max = (now + timedelta(days=7)).isoformat()

    params: dict[str, Any] = {
        "timeMin": time_min,
        "timeMax": time_max,
        "maxResults": min(max(1, max_results), 100),
        "singleEvents": "true",
        "orderBy": "startTime",
    }
    if query:
        params["q"] = query

    result = await _gcal_request(
        "GET", f"{GCAL_BASE}/calendars/{calendar_id}/events", params=params,
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def create_event(
    summary: str,
    start: str,
    end: str,
    description: str = "",
    location: str = "",
    attendees: str = "",
    calendar_id: str = "primary",
    timezone: str = "",
    add_meet: bool = False,
) -> str:
    """Create a new calendar event.

    Args:
        summary: Event title.
        start: Start time in ISO 8601 format (e.g. "2025-01-15T10:00:00+02:00").
                For all-day events use date format: "2025-01-15".
        end: End time in ISO 8601 format.
             For all-day events use the day after: "2025-01-16".
        description: Event description/notes (optional).
        location: Event location (optional).
        attendees: Comma-separated email addresses of attendees (optional).
        calendar_id: Calendar ID (default "primary").
        timezone: IANA timezone for the event (e.g. "Europe/Bucharest"). Optional.
        add_meet: If true, adds a Google Meet video conference link.
    """
    is_all_day = "T" not in start

    body: dict[str, Any] = {"summary": summary}

    if is_all_day:
        body["start"] = {"date": start}
        body["end"] = {"date": end}
    else:
        body["start"] = {"dateTime": start}
        body["end"] = {"dateTime": end}
        if timezone:
            body["start"]["timeZone"] = timezone
            body["end"]["timeZone"] = timezone

    if description:
        body["description"] = description
    if location:
        body["location"] = location
    if attendees:
        body["attendees"] = [
            {"email": e.strip()} for e in attendees.split(",") if e.strip()
        ]
    if add_meet:
        body["conferenceData"] = {
            "createRequest": {"requestId": f"meet-{datetime.now().timestamp()}"}
        }

    params = {}
    if add_meet:
        params["conferenceDataVersion"] = 1

    result = await _gcal_request(
        "POST",
        f"{GCAL_BASE}/calendars/{calendar_id}/events",
        params=params,
        json_body=body,
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def update_event(
    event_id: str,
    summary: str = "",
    start: str = "",
    end: str = "",
    description: str = "",
    location: str = "",
    calendar_id: str = "primary",
) -> str:
    """Update an existing calendar event.

    Only provided fields are changed — omitted fields keep their current values.
    Use list_events to find the event_id.

    Args:
        event_id: The event ID to update.
        summary: New event title (optional).
        start: New start time in ISO 8601 format (optional).
        end: New end time in ISO 8601 format (optional).
        description: New description (optional).
        location: New location (optional).
        calendar_id: Calendar ID (default "primary").
    """
    body: dict[str, Any] = {}
    if summary:
        body["summary"] = summary
    if start:
        if "T" not in start:
            body["start"] = {"date": start}
        else:
            body["start"] = {"dateTime": start}
    if end:
        if "T" not in end:
            body["end"] = {"date": end}
        else:
            body["end"] = {"dateTime": end}
    if description:
        body["description"] = description
    if location:
        body["location"] = location

    if not body:
        return json.dumps({"error": "No fields to update"})

    result = await _gcal_request(
        "PATCH",
        f"{GCAL_BASE}/calendars/{calendar_id}/events/{event_id}",
        json_body=body,
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def delete_event(
    event_id: str,
    calendar_id: str = "primary",
) -> str:
    """Delete a calendar event.

    Args:
        event_id: The event ID to delete.
        calendar_id: Calendar ID (default "primary").
    """
    result = await _gcal_request(
        "DELETE",
        f"{GCAL_BASE}/calendars/{calendar_id}/events/{event_id}",
    )
    return json.dumps(result, indent=2)


# -- Tasks tools --------------------------------------------------------------


@mcp.tool()
async def list_task_lists() -> str:
    """List all task lists for the user.

    Returns task list IDs and titles.  Use the task list ID in other
    task tools (default is "@default").
    """
    result = await _gcal_request("GET", f"{GTASKS_BASE}/users/@me/lists")
    return json.dumps(result, indent=2)


@mcp.tool()
async def list_tasks(
    tasklist_id: str = "@default",
    show_completed: bool = False,
    max_results: int = 50,
) -> str:
    """List tasks in a task list.

    Args:
        tasklist_id: Task list ID (default "@default" for the primary list).
        show_completed: Include completed tasks (default false).
        max_results: Maximum number of tasks to return (1-100).
    """
    params: dict[str, Any] = {
        "maxResults": min(max(1, max_results), 100),
        "showCompleted": str(show_completed).lower(),
    }
    result = await _gcal_request(
        "GET", f"{GTASKS_BASE}/lists/{tasklist_id}/tasks", params=params,
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def create_task(
    title: str,
    notes: str = "",
    due: str = "",
    tasklist_id: str = "@default",
) -> str:
    """Create a new task.

    Args:
        title: Task title.
        notes: Task notes/description (optional).
        due: Due date in ISO 8601 format, e.g. "2025-01-15T00:00:00Z" (optional).
        tasklist_id: Task list ID (default "@default").
    """
    body: dict[str, Any] = {"title": title}
    if notes:
        body["notes"] = notes
    if due:
        body["due"] = due

    result = await _gcal_request(
        "POST", f"{GTASKS_BASE}/lists/{tasklist_id}/tasks", json_body=body,
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def complete_task(
    task_id: str,
    tasklist_id: str = "@default",
) -> str:
    """Mark a task as completed.

    Args:
        task_id: The task ID to complete.
        tasklist_id: Task list ID (default "@default").
    """
    result = await _gcal_request(
        "PATCH",
        f"{GTASKS_BASE}/lists/{tasklist_id}/tasks/{task_id}",
        json_body={"status": "completed"},
    )
    return json.dumps(result, indent=2)


# -- Main --------------------------------------------------------------------

if __name__ == "__main__":
    _load_tokens()
    mcp.run(transport="stdio")
