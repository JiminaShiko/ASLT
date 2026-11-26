"""
Pusher Channels for the admin session monitor.

The admin page subscribes to one private channel and refreshes when the server says a
session started, ended, closed or was revoked. This module only needs the server SDK;
when PUSHER_* is not configured it does nothing and the page keeps its 30 s polling.

Never exposed: PUSHER_APP_SECRET. public_config() returns only the browser-safe key.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone

log = logging.getLogger(__name__)

DEFAULT_CHANNEL = "private-admin-sessions"
EVENT_NAME = "sessions-changed"
AUTH_ENDPOINT = "/api/realtime/auth"

_client = None
_client_loaded = False
_client_lock = threading.Lock()


def _setting(name: str) -> str:
    return (os.getenv(name) or "").strip()


def is_configured() -> bool:
    return all(
        _setting(name)
        for name in ("PUSHER_APP_ID", "PUSHER_APP_KEY", "PUSHER_APP_SECRET", "PUSHER_APP_CLUSTER")
    )


def channel_name() -> str:
    return _setting("PUSHER_CHANNEL") or DEFAULT_CHANNEL


def _get_client():
    """The Pusher client, or None when the app has no Pusher credentials."""
    global _client, _client_loaded
    if _client_loaded:
        return _client
    with _client_lock:
        if _client_loaded:
            return _client
        _client_loaded = True
        if not is_configured():
            return None
        try:
            import pusher  # imported lazily so a missing package never breaks startup

            _client = pusher.Pusher(
                app_id=_setting("PUSHER_APP_ID"),
                key=_setting("PUSHER_APP_KEY"),
                secret=_setting("PUSHER_APP_SECRET"),
                cluster=_setting("PUSHER_APP_CLUSTER"),
                ssl=True,
                timeout=3,
            )
        except Exception:  # noqa: BLE001 - realtime must never take the app down
            log.exception("Pusher client could not be created; session updates stay on polling")
            _client = None
    return _client


def public_config() -> dict:
    """What the browser needs to connect. The secret is never included."""
    if not is_configured():
        return {"enabled": False}
    return {
        "enabled": True,
        "key": _setting("PUSHER_APP_KEY"),
        "cluster": _setting("PUSHER_APP_CLUSTER"),
        "channel": channel_name(),
        "event": EVENT_NAME,
        "authEndpoint": AUTH_ENDPOINT,
    }


def authenticate(channel: str, socket_id: str):
    """Private-channel authorization for one subscription (computed locally by the SDK)."""
    client = _get_client()
    if client is None:
        return None
    return client.authenticate(channel, socket_id)


def publish_sessions_changed(reason: str) -> None:
    """Tell connected admin pages to refresh. Fire-and-forget: never blocks or raises."""
    client = _get_client()
    if client is None:
        return
    payload = {"reason": reason, "at": datetime.now(timezone.utc).isoformat()}

    def _send():
        try:
            client.trigger(channel_name(), EVENT_NAME, payload)
        except Exception:  # noqa: BLE001 - a missed event only means the page polls
            log.warning("Pusher event could not be delivered; admin pages will poll", exc_info=True)

    threading.Thread(target=_send, name="pusher-sessions", daemon=True).start()
