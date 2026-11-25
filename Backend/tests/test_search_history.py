import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from flask import Flask

from user.search_history import history_bp


class FakeCursor:
    def __init__(self):
        self.calls = []
        self.rows = []
        self.lastrowid = 17
        self.media_path = None
        self.closed = False

    def execute(self, query, params=None):
        self.calls.append((query, params))
        if "SELECT search_id, search_keyword" in query:
            self.rows = [
                (17, "hello", "text", '{"items":[]}', None, None, None)
            ]
        elif "INSERT INTO search_history" in query:
            self.media_path = params[4]
        elif "SELECT media_path, media_type" in query:
            self.rows = [(self.media_path, "image")]

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self):
        self.cursor_instance = FakeCursor()
        self.closed = False
        self.committed = False

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.committed = True

    def rollback(self):
        pass

    def close(self):
        self.closed = True


class SearchHistoryTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.secret_key = "test-secret"
        self.app.register_blueprint(history_bp)
        self.client = self.app.test_client()

    def authenticate(self, user_id=42):
        with self.client.session_transaction() as browser_session:
            browser_session["user_id"] = user_id
            browser_session["user_email"] = "user@example.test"

    def test_history_requires_authentication(self):
        response = self.client.get("/api/search-history")
        self.assertEqual(response.status_code, 401)

    @patch("user.search_history.get_db_connection")
    def test_list_returns_only_signed_in_users_records(self, get_connection):
        connection = FakeConnection()
        get_connection.return_value = connection
        self.authenticate()

        response = self.client.get("/api/search-history")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["items"][0]["search_keyword"], "hello")
        query, params = connection.cursor_instance.calls[0]
        self.assertIn("WHERE user_id = %s", query)
        self.assertEqual(params, (42,))
        self.assertTrue(connection.closed)

    @patch("user.search_history.get_db_connection")
    def test_create_history_uses_session_user_and_returns_saved_entry(self, get_connection):
        connection = FakeConnection()
        get_connection.return_value = connection
        self.authenticate()

        response = self.client.post(
            "/api/search-history",
            json={
                "search_type": "webcam_letters",
                "search_keyword": "",
                "search_result": "hello",
                "user_id": 999,
            },
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_json(), {"search_id": 17})
        insert = connection.cursor_instance.calls[-1]
        self.assertIn("INSERT INTO search_history", insert[0])
        self.assertEqual(insert[1][0], 42)
        self.assertEqual(insert[1][2], "webcam_letters")
        self.assertTrue(connection.committed)
        self.assertTrue(connection.closed)

    @patch("user.search_history.get_db_connection")
    def test_upload_history_stores_media_and_returns_owner_scoped_media(self, get_connection):
        connection = FakeConnection()
        get_connection.return_value = connection
        self.authenticate()

        with tempfile.TemporaryDirectory() as temp_dir:
            media_dir = Path(temp_dir)
            with patch("user.search_history.MEDIA_DIR", media_dir):
                response = self.client.post(
                    "/api/search-history",
                    data={
                        "search_type": "upload",
                        "search_keyword": "sign.png",
                        "search_result": "hello",
                        "media": (io.BytesIO(b"image-bytes"), "sign.png", "image/png"),
                    },
                    content_type="multipart/form-data",
                )
                self.assertEqual(response.status_code, 201)
                stored_file = next(media_dir.iterdir())
                self.assertEqual(stored_file.read_bytes(), b"image-bytes")

                media_response = self.client.get(
                    "/api/search-history/17/media"
                )
                self.assertEqual(media_response.status_code, 200)
                self.assertEqual(media_response.data, b"image-bytes")
                media_response.close()

        insert = connection.cursor_instance.calls[0]
        self.assertEqual(insert[1][0], 42)
        self.assertEqual(insert[1][2], "upload")

    def test_rejects_unknown_history_type(self):
        self.authenticate()
        response = self.client.post(
            "/api/search-history",
            json={"search_type": "other", "search_result": "hello"},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
