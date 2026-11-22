import unittest
from datetime import datetime
from unittest.mock import patch

from flask import Flask

from admin.profile import admin_profile_bp


class FakeCursor:
    def __init__(self):
        self.closed = False
        self.dictionary = False

    def execute(self, query, params):
        self.params = params

    def fetchone(self):
        return {
            "user_id": 7,
            "username": "admin",
            "email": "admin@example.test",
            "full_name": "Admin User",
            "profile_photo": "/uploads/profile_photos/admin.png",
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

    def cursor(self, dictionary=False):
        self.cursor_instance.dictionary = dictionary
        return self.cursor_instance

    def close(self):
        self.closed = True


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
                "full_name": "Admin User",
                "profile_photo": "/uploads/profile_photos/admin.png",
                "role": "admin",
                "status": "active",
                "created_at": "2025-01-01T00:00:00",
                "updated_at": "2026-11-22T00:00:00",
            },
        )
        self.assertTrue(connection.cursor_instance.dictionary)
        self.assertEqual(connection.cursor_instance.params, (7, "admin@example.test"))
        self.assertTrue(connection.cursor_instance.closed)
        self.assertTrue(connection.closed)


if __name__ == "__main__":
    unittest.main()
