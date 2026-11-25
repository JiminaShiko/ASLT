import unittest
from unittest.mock import patch

from flask import Flask, g

from admin.quiz_router import admin_quiz_bp


class FakeCursor:
    def __init__(self):
        self.questions = []
        self.options = []
        self.result = []
        self.lastrowid = None
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def execute(self, query, params=None):
        query = query.strip()
        params = params or ()
        if query.startswith("SELECT material_id, sign_label FROM sign_materials"):
            self.result = [
                {"material_id": material_id, "sign_label": f"Sign {material_id}"}
                for material_id in params
            ]
        elif query.startswith("INSERT INTO admin_quiz_questions"):
            question_id = len(self.questions) + 1
            self.lastrowid = question_id
            self.questions.append(
                {
                    "question_id": question_id,
                    "question_type": params[0],
                    "prompt_material_id": params[1],
                    "prompt_text": params[2],
                    "category": params[3],
                    "status": params[4],
                    "correct_material_id": params[5],
                }
            )
        elif query.startswith("DELETE FROM admin_quiz_options"):
            self.options = [
                row for row in self.options if row[0] != params[0]
            ]
        elif query.startswith("SELECT question_id, question_type"):
            self.result = list(self.questions)
        elif query.startswith("SELECT question_id, material_id FROM admin_quiz_options"):
            question_ids = params
            self.result = [
                {"question_id": question_id, "material_id": material_id}
                for question_id, material_id, _position in self.options
                if question_id in question_ids
            ]
        else:
            raise AssertionError(f"Unexpected query: {query}")

    def executemany(self, query, rows):
        if not query.startswith("INSERT INTO admin_quiz_options"):
            raise AssertionError(f"Unexpected query: {query}")
        self.options.extend(rows)

    def fetchone(self):
        return self.result[0] if self.result else None

    def fetchall(self):
        return self.result

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self):
        self.cursor_instance = FakeCursor()
        self.closed = False
        self.committed = False

    def cursor(self, **_kwargs):
        return self.cursor_instance

    def commit(self):
        self.committed = True

    def rollback(self):
        pass

    def close(self):
        self.closed = True


class AdminQuizRouterTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.secret_key = "test-secret"
        self.app.register_blueprint(admin_quiz_bp)

        @self.app.before_request
        def set_admin_context():
            g.signrest_account = {"role": "admin", "status": "active"}

        self.client = self.app.test_client()

    def test_questions_requires_an_admin_session(self):
        response = self.client.get("/api/admin/quiz/questions")
        self.assertEqual(response.status_code, 401)

    @patch("admin.quiz_router.get_db_connection")
    def test_create_question_persists_admin_question_and_ordered_options(self, get_connection):
        connection = FakeConnection()
        get_connection.return_value = connection
        with self.client.session_transaction() as browser_session:
            browser_session["user_id"] = 9
            browser_session["user_email"] = "admin@example.test"
            browser_session["user_role"] = "admin"

        response = self.client.post(
            "/api/admin/quiz/questions",
            json={
                "type": "sign_to_text",
                "prompt_material_id": 1,
                "prompt_text": "",
                "category": "Alphabet",
                "status": "active",
                "options": [1, 2, 3, 4],
                "correct": 1,
            },
            headers={"X-Signrest-Request": "admin"},
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(
            response.get_json()["item"],
            {
                "id": 1,
                "type": "sign_to_text",
                "prompt_material_id": 1,
                "prompt_text": "",
                "category": "Alphabet",
                "status": "active",
                "options": [1, 2, 3, 4],
                "correct": 1,
            },
        )
        self.assertEqual(connection.cursor_instance.options, [(1, 1, 0), (1, 2, 1), (1, 3, 2), (1, 4, 3)])
        self.assertTrue(connection.committed)
        self.assertTrue(connection.closed)

    @patch("admin.quiz_router.get_db_connection")
    def test_create_rejects_repeated_options_without_committing(self, get_connection):
        connection = FakeConnection()
        get_connection.return_value = connection
        with self.client.session_transaction() as browser_session:
            browser_session["user_id"] = 9
            browser_session["user_email"] = "admin@example.test"
            browser_session["user_role"] = "admin"

        response = self.client.post(
            "/api/admin/quiz/questions",
            json={
                "type": "sign_to_text",
                "prompt_material_id": 1,
                "category": "Alphabet",
                "status": "active",
                "options": [1, 1, 2, 3],
                "correct": 1,
            },
            headers={"X-Signrest-Request": "admin"},
        )

        self.assertEqual(response.status_code, 422)
        self.assertFalse(connection.committed)
        self.assertEqual(connection.cursor_instance.questions, [])


if __name__ == "__main__":
    unittest.main()
