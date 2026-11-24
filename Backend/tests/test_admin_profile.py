import unittest
from datetime import datetime
from io import BytesIO
from unittest.mock import patch

from flask import Flask

from admin.profile import admin_profile_bp


class FakeCursor:
    def __init__(self):
        self.closed = False
        self.dictionary = False
        self.query = ""
        self.executions = []

    def execute(self, query, params):
        self.query = query
        self.params = params
        self.executions.append((query, params))

    def fetchone(self):
        if "WHERE username = %s" in self.query:
            return None
        return {
            "user_id": 7,
            "username": "admin",
            "email": "admin@example.test",
            "profile_photo": "/uploads/profile_photos/admin.png",
            "password_hash": "existing-password-hash",
            "role": "admin",
            "status": "active",
            "created_at": datetime(2025, 1, 1),
            "updated_at": datetime(2026, 11, 22),
        }

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self):
        self.cursor_instance = FakeCursor()
        self.closed = False
        self.committed = False

    def cursor(self, dictionary=False):
        self.cursor_instance.dictionary = dictionary
        return self.cursor_instance

    def close(self):
        self.closed = True

    def commit(self):
        self.committed = True

    def rollback(self):
        pass


class AdminProfileTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.secret_key = "test-secret"
        self.app.register_blueprint(admin_profile_bp)
        self.client = self.app.test_client()

    def test_requires_an_admin_session(self):
        response = self.client.get("/api/admin/profile")
        self.assertEqual(response.status_code, 401)

        with self.client.session_transaction() as browser_session:
            browser_session["user_id"] = 8
            browser_session["user_email"] = "user@example.test"
            browser_session["user_role"] = "user"
        response = self.client.get("/api/admin/profile")
        self.assertEqual(response.status_code, 403)

    @patch("admin.profile.get_db_connection")
    def test_returns_profile_fields_for_logged_in_admin(self, get_connection):
        connection = FakeConnection()
        get_connection.return_value = connection
        with self.client.session_transaction() as browser_session:
            browser_session["user_id"] = 7
            browser_session["user_email"] = "admin@example.test"
            browser_session["user_role"] = "admin"

        response = self.client.get("/api/admin/profile")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["item"],
            {
                "user_id": 7,
                "username": "admin",
                "email": "admin@example.test",
                "profile_photo": "/uploads/profile_photos/admin.png",
                "role": "admin",
                "status": "active",
                "created_at": "2025-01-01T00:00:00",
                "updated_at": "2026-11-22T00:00:00",
            },
        )
        self.assertTrue(connection.cursor_instance.dictionary)
        self.assertNotIn("full_name", connection.cursor_instance.query.lower())
        self.assertEqual(connection.cursor_instance.params, (7, "admin@example.test"))
        self.assertTrue(connection.cursor_instance.closed)
        self.assertTrue(connection.closed)

    @patch("admin.profile.get_db_connection")
    def test_updates_username_using_existing_users_table_columns(self, get_connection):
        connection = FakeConnection()
        get_connection.return_value = connection
        with self.client.session_transaction() as browser_session:
            browser_session["user_id"] = 7
            browser_session["user_email"] = "admin@example.test"
            browser_session["user_role"] = "admin"

        response = self.client.put(
            "/api/admin/profile",
            json={"username": "new_admin"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["item"],
            {"username": "new_admin"},
        )
        self.assertEqual(connection.cursor_instance.executions[-1][1][0], "new_admin")
        self.assertNotIn("full_name", connection.cursor_instance.query.lower())
        self.assertTrue(connection.committed)

    @patch("admin.profile.upload_to_cloudinary", return_value="https://example.test/avatar.png")
    @patch("admin.profile.get_db_connection")
    def test_profile_picture_upload_saves_returned_url(self, get_connection, _upload):
        connection = FakeConnection()
        get_connection.return_value = connection
        with self.client.session_transaction() as browser_session:
            browser_session["user_id"] = 7
            browser_session["user_email"] = "admin@example.test"
            browser_session["user_role"] = "admin"

        response = self.client.post(
            "/api/admin/profile/upload-picture",
            data={"profile_picture": (BytesIO(b"\x89PNG\r\n\x1a\n"), "avatar.png")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["profile_photo"], "https://example.test/avatar.png")
        self.assertEqual(
            connection.cursor_instance.executions[-1][1][0],
            "https://example.test/avatar.png",
        )
        self.assertTrue(connection.committed)

    @patch("admin.profile.get_db_connection")
    @patch("admin.profile.check_password_hash", return_value=True)
    def test_password_change_accepts_confirmed_password_and_commits(self, _check_hash, get_connection):
        connection = FakeConnection()
        get_connection.return_value = connection
        with self.client.session_transaction() as browser_session:
            browser_session["user_id"] = 7
            browser_session["user_email"] = "admin@example.test"
            browser_session["user_role"] = "admin"

        response = self.client.post(
            "/api/admin/profile/change-password",
            json={
                "current_password": "current-pass",
                "new_password": "Newpass1!",
                "confirm_password": "Newpass1!",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["success"], True)
        self.assertIn("SET password_hash = %s", connection.cursor_instance.query)
        self.assertTrue(connection.committed)


if __name__ == "__main__":
    unittest.main()
