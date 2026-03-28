"""OAuth2 flow handler for connector integrations.

Manages the browser-based OAuth2 authorization code flow:
  1. Build an auth URL and return it to the frontend
  2. Spin up a temporary HTTP server to receive the callback
  3. Exchange the authorization code for tokens
  4. Store tokens via CredentialStore
  5. Restart the connector if it was running

Currently supports Google OAuth2 (Calendar + Tasks).  The pattern is
generic enough to support other OAuth2 providers in the future.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.parse
import urllib.request
from typing import Any

log = logging.getLogger(__name__)

# -- Google OAuth2 constants --------------------------------------------------

GOOGLE_CLIENT_ID = os.environ.get(
    "GOOGLE_CLIENT_ID",
    "",
)
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"

GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/tasks",
]

# Port for the local OAuth callback server
OAUTH_CALLBACK_PORT = 8420
OAUTH_REDIRECT_URI = f"http://localhost:{OAUTH_CALLBACK_PORT}"


# -- OAuth2 integration registry ---------------------------------------------

# Maps connector_id -> OAuth config.  Extensible for future providers.
_OAUTH_CONFIGS: dict[str, dict[str, Any]] = {
    "google_calendar": {
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "auth_endpoint": GOOGLE_AUTH_ENDPOINT,
        "token_endpoint": GOOGLE_TOKEN_ENDPOINT,
        "scopes": GOOGLE_SCOPES,
        "redirect_uri": OAUTH_REDIRECT_URI,
    },
}


def get_oauth_config(connector_id: str) -> dict[str, Any] | None:
    """Return OAuth config for a connector, or None if it doesn't use OAuth."""
    return _OAUTH_CONFIGS.get(connector_id)


def requires_oauth(connector_id: str) -> bool:
    """Check if a connector requires OAuth authentication."""
    return connector_id in _OAUTH_CONFIGS


# -- Auth URL builder ---------------------------------------------------------

def build_auth_url(connector_id: str) -> str:
    """Build the OAuth2 authorization URL for a connector.

    Returns the full URL the user should open in their browser.
    """
    config = _OAUTH_CONFIGS.get(connector_id)
    if config is None:
        raise ValueError(f"No OAuth config for connector '{connector_id}'")

    params = {
        "client_id": config["client_id"],
        "redirect_uri": config["redirect_uri"],
        "response_type": "code",
        "scope": " ".join(config["scopes"]),
        "access_type": "offline",
        "prompt": "consent",
    }
    return f"{config['auth_endpoint']}?{urllib.parse.urlencode(params)}"


# -- Token exchange -----------------------------------------------------------

def exchange_code(connector_id: str, code: str) -> dict[str, Any]:
    """Exchange an authorization code for access + refresh tokens.

    Returns the token response dict from the provider.
    Raises on failure.
    """
    config = _OAUTH_CONFIGS.get(connector_id)
    if config is None:
        raise ValueError(f"No OAuth config for connector '{connector_id}'")

    data = urllib.parse.urlencode({
        "client_id": config["client_id"],
        "client_secret": config["client_secret"],
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": config["redirect_uri"],
    }).encode()

    req = urllib.request.Request(config["token_endpoint"], data=data, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


# -- Callback HTTP server -----------------------------------------------------

_SUCCESS_HTML = """\
<!DOCTYPE html>
<html>
<head><title>ATN — Authorization Complete</title></head>
<body style="font-family: system-ui; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background: #0a0a0a; color: #e0e0e0;">
<div style="text-align: center;">
<h1 style="color: #4ade80;">&#10003; Connected</h1>
<p>Google Calendar is now connected to ATN.</p>
<p style="color: #888;">You can close this tab.</p>
</div>
</body>
</html>"""

_ERROR_HTML = """\
<!DOCTYPE html>
<html>
<head><title>ATN — Authorization Failed</title></head>
<body style="font-family: system-ui; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background: #0a0a0a; color: #e0e0e0;">
<div style="text-align: center;">
<h1 style="color: #ef4444;">&#10007; Failed</h1>
<p>Authorization failed: {error}</p>
<p style="color: #888;">Close this tab and try again.</p>
</div>
</body>
</html>"""


async def run_oauth_callback(
    connector_id: str,
    *,
    timeout: float = 120,
) -> dict[str, Any]:
    """Start a temporary HTTP server to receive the OAuth callback.

    Waits for Google to redirect with ?code=... or ?error=..., then
    exchanges the code for tokens and returns them.

    Returns the token dict (access_token, refresh_token, etc.)
    Raises on timeout or auth error.
    """
    result_future: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()

    async def handle_request(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=5)
            request_str = request_line.decode("utf-8", errors="replace")

            # Parse GET /?code=...&scope=... HTTP/1.1
            path = request_str.split(" ")[1] if " " in request_str else "/"
            query = urllib.parse.urlparse(path).query
            params = urllib.parse.parse_qs(query)

            code = params.get("code", [None])[0]
            error = params.get("error", [None])[0]

            if error:
                html = _ERROR_HTML.format(error=error)
                writer.write(f"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {len(html)}\r\nConnection: close\r\n\r\n{html}".encode())
                await writer.drain()
                writer.close()
                if not result_future.done():
                    result_future.set_exception(RuntimeError(f"OAuth error: {error}"))
                return

            if not code:
                html = _ERROR_HTML.format(error="no authorization code received")
                writer.write(f"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {len(html)}\r\nConnection: close\r\n\r\n{html}".encode())
                await writer.drain()
                writer.close()
                if not result_future.done():
                    result_future.set_exception(RuntimeError("No authorization code in callback"))
                return

            # Exchange code for tokens
            try:
                tokens = exchange_code(connector_id, code)
            except Exception as exc:
                html = _ERROR_HTML.format(error=str(exc))
                writer.write(f"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {len(html)}\r\nConnection: close\r\n\r\n{html}".encode())
                await writer.drain()
                writer.close()
                if not result_future.done():
                    result_future.set_exception(exc)
                return

            html = _SUCCESS_HTML
            writer.write(f"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {len(html)}\r\nConnection: close\r\n\r\n{html}".encode())
            await writer.drain()
            writer.close()

            if not result_future.done():
                result_future.set_result(tokens)

        except Exception as exc:
            writer.close()
            if not result_future.done():
                result_future.set_exception(exc)

    server = await asyncio.start_server(handle_request, "localhost", OAUTH_CALLBACK_PORT)

    try:
        tokens = await asyncio.wait_for(result_future, timeout=timeout)
        return tokens
    except asyncio.TimeoutError:
        raise RuntimeError(f"OAuth callback timed out after {timeout}s")
    finally:
        server.close()
        await server.wait_closed()
