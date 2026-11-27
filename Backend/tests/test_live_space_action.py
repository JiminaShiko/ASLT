import threading
import unittest

from flask import Flask

from user import api


def bare_session(text=""):
    """LiveSession.action only touches the text buffer, not the model."""
    session = object.__new__(api.LiveSession)
    session.lock = threading.RLock()
    session.text = text
    session.text_version = 0
    session.last_activity = 0.0
    session.recent_commits = []
    session.commit_notes = []
    session.response = lambda: {
        "text": session.text,
        "text_version": session.text_version,
    }
    return session


class LiveSpaceActionTests(unittest.TestCase):
    def test_space_appends_and_advances_text_version(self):
        session = bare_session("AB")

        first = session.action("space")
        self.assertEqual(first["text"], "AB ")

        second = session.action("space")
        self.assertEqual(second["text"], "AB  ")
        self.assertEqual(second["text_version"], 2)

    def test_backspace_over_space_keeps_letter_notes(self):
        session = bare_session("AB")
        session.recent_commits = ["A", "B"]
        session.commit_notes = ["note-A", "note-B"]

        session.action("space")
        session.action("backspace")

        self.assertEqual(session.text, "AB")
        self.assertEqual(session.recent_commits, ["A", "B"])
        self.assertEqual(session.commit_notes, ["note-A", "note-B"])


class LiveActionRouteTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.register_blueprint(api.api_bp)
        self.client = self.app.test_client()

    def test_space_route_returns_typed_text_with_space(self):
        session = bare_session("HI")
        with api.SESSIONS_LOCK:
            api.SESSIONS["test-space-action"] = session
        try:
            response = self.client.post(
                "/api/live/action",
                json={"session_id": "test-space-action", "action": "space"},
            )
        finally:
            with api.SESSIONS_LOCK:
                api.SESSIONS.pop("test-space-action", None)

        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertEqual(response.get_json()["text"], "HI ")
