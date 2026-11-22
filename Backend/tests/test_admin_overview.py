import unittest
from datetime import date, datetime, timedelta
from unittest.mock import patch

from flask import Flask

from admin.overview import admin_overview_bp


class FakeCursor:
    def __init__(self, activity_date):
        self.activity_date = activity_date
        self.result = None
        self.closed = False
        self.calls = []

    def execute(self, query, params=None):
        self.calls.append((query, params))
        if "COUNT(*) FROM users" in query:
            self.result = (4,)
        elif "COUNT(*) FROM translations" in query:
            self.result = (8,)
        elif "COUNT(*) FROM user_sessions" in query:
            self.result = (2,)
        elif "COUNT(*) FROM feedbacks" in query:
            self.result = (1,)
        elif "FROM model_accuracy" in query:
            self.result = (
                "Signnet",
                "v1",
                100,
                96.5,
                datetime(2026, 11, 20, 12, 0),
            )
        elif "GROUP BY DATE(created_at), mode" in query:
            self.result = [
                (self.activity_date, "sign_to_text", 3),
                (self.activity_date, "text_to_sign", 2),
            ]
        elif "WHERE t.confidence_score < 75" in query:
            self.result = [
                (
                    5,
                    "Alex",
                    "sign_to_text",
                    "webcam",
                    "hello",
                    72.5,
                    datetime(2026, 11, 20, 12, 0),
                )
            ]
        elif "FROM feedbacks f" in query:
            self.result = [
                (
                    9,
                    "Sam",
                    5,
                    "Helpful",
                    "pending",
                    datetime(2026, 11, 20, 12, 0),
                )
            ]
        else:
            raise AssertionError("Unexpected query: " + query)

    def fetchone(self):
        return self.result

    def fetchall(self):
        return self.result

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self):
        self.cursor_instance = FakeCursor(date.today() - timedelta(days=6))
        self.closed = False

    def cursor(self):
        return self.cursor_instance

    def close(self):
        self.closed = True


class AdminOverviewTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.secret_key = "test-secret"
        self.app.register_blueprint(admin_overview_bp)
        self.client = self.app.test_client()

    def test_requires_an_admin_session(self):
        response = self.client.get("/api/admin/overview")
        self.assertEqual(response.status_code, 401)

        with self.client.session_transaction() as browser_session:
            browser_session["user_email"] = "user@example.test"
            browser_session["user_role"] = "user"
        response = self.client.get("/api/admin/overview")
        self.assertEqual(response.status_code, 403)

    @patch("admin.overview.get_db_connection")
    def test_returns_expected_overview_payload(self, get_connection):
        connection = FakeConnection()
        get_connection.return_value = connection
        with self.client.session_transaction() as browser_session:
            browser_session["user_email"] = "admin@example.test"
            browser_session["user_role"] = "admin"
            browser_session["username"] = "Admin"

        response = self.client.get("/api/admin/overview?days=7")

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["admin_name"], "Admin")
        self.assertEqual(
            data["totals"],
            {
                "users": 4,
                "translations": 8,
                "active_sessions": 2,
                "pending_feedback": 1,
            },
        )
        self.assertEqual(len(data["activity"]), 7)
        self.assertEqual(data["activity"][0]["sign_to_text"], 3)
        self.assertEqual(data["activity"][0]["text_to_sign"], 2)
        self.assertEqual(data["model"]["accuracy_percentage"], 96.5)
        self.assertEqual(data["low_confidence"][0]["mode"], "Sign to Text")
        self.assertEqual(data["recent_feedback"][0]["user_name"], "Sam")
        self.assertTrue(
            any(
                "status IN" in query and params == ("pending", "new")
                for query, params in connection.cursor_instance.calls
            )
        )
        self.assertTrue(connection.cursor_instance.closed)
        self.assertTrue(connection.closed)

    @patch("admin.overview.get_db_connection")
    def test_database_failure_returns_error_instead_of_empty_data(self, get_connection):
        get_connection.side_effect = RuntimeError("database unavailable")
        with self.client.session_transaction() as browser_session:
            browser_session["user_email"] = "admin@example.test"
            browser_session["user_role"] = "admin"

        response = self.client.get("/api/admin/overview")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json(), {"message": "Could not load overview"})


if __name__ == "__main__":
    unittest.main()
