import unittest
from unittest.mock import patch

from flask import Flask

from user.learning import learning_bp


class Cursor:
    def __init__(self, attempts, status="active"):
        self.attempts = attempts
        self.status = status
        self.queries = []
        self.result = None
        self.questions = [(10, "A", "multiple_choice", '["A", "B"]')]

    def execute(self, sql, params=None):
        sql = " ".join(sql.split())
        self.queries.append(sql)
        if sql.startswith("SELECT user_id FROM users"):
            self.result = (1,)
        elif sql.startswith("SELECT status FROM quizzes"):
            self.result = (self.status,)
        elif sql.startswith("SELECT COUNT(*) FROM quiz_results"):
            self.result = (self.attempts,)
        elif sql.startswith("SELECT COUNT(*) FROM quiz_attempts"):
            self.result = (0,)
        elif sql.startswith("SELECT question_id, correct_answer"):
            self.result = self.questions
        elif sql.startswith("INSERT INTO quiz_results"):
            self.result = None
        else:
            raise AssertionError(sql)

    def fetchone(self):
        return self.result

    def fetchall(self):
        return self.result

    def close(self):
        pass


class Connection:
    def __init__(self, attempts, status="active"):
        self.cur = Cursor(attempts, status)
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self.cur

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        pass


class QuizAttemptLimitTests(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.secret_key = "test"
        app.register_blueprint(learning_bp)
        self.client = app.test_client()
        with self.client.session_transaction() as session:
            session["user_id"] = 1

    def submit(self, connection):
        with patch("user.learning.get_db_connection", return_value=connection):
            return self.client.post("/api/quizzes/7/submit", json={"answers": {"10": "A"}})

    def test_third_attempt_is_saved(self):
        conn = Connection(2)
        response = self.submit(conn)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["score"], 1)
        self.assertTrue(conn.committed)
        self.assertTrue(any("FOR UPDATE" in sql for sql in conn.cur.queries))

    def test_fourth_attempt_is_rejected(self):
        conn = Connection(3)
        response = self.submit(conn)
        self.assertEqual(response.status_code, 409)
        self.assertTrue(conn.rolled_back)
        self.assertFalse(any(sql.startswith("INSERT") for sql in conn.cur.queries))

    def test_unpublished_quiz_is_rejected(self):
        conn = Connection(0, "draft")
        response = self.submit(conn)
        self.assertEqual(response.status_code, 404)
        self.assertTrue(conn.rolled_back)

    def test_incomplete_quiz_does_not_consume_attempt(self):
        conn = Connection(0)
        with patch("user.learning.get_db_connection", return_value=conn):
            response = self.client.post("/api/quizzes/7/submit", json={"answers": {}})
        self.assertEqual(response.status_code, 400)
        self.assertTrue(conn.rolled_back)
        self.assertFalse(any(sql.startswith("INSERT") for sql in conn.cur.queries))

    def test_new_format_requires_the_saved_attempt_flow(self):
        conn = Connection(0)
        conn.cur.questions = [(10, "0", "multiple_choice", '[{"type":"image","value":"/assets/a.png"},{"type":"text","value":"B"}]')]
        with patch("user.learning.get_db_connection", return_value=conn):
            response = self.client.post("/api/quizzes/7/submit", json={"answers": {"10": "0"}})
        self.assertEqual(response.status_code, 409)
        self.assertFalse(conn.committed)

    def test_camera_questions_cannot_be_submitted_as_text(self):
        conn = Connection(0)
        conn.cur.questions = [(10, "", "camera", "[]")]
        response = self.submit(conn)
        self.assertEqual(response.status_code, 409)
        self.assertFalse(conn.committed)

    def test_rank_uses_best_quiz_attempt_and_counts_uncompleted_quizzes(self):
        class RankCursor:
            def execute(self, sql, params=None):
                sql = " ".join(sql.split())
                if sql.startswith("SELECT user_id FROM users WHERE role"):
                    self.rows = [(1,), (2,)]
                elif sql.startswith("SELECT user_id, category, item_key"):
                    self.rows = [(1, "asl", "A"), (2, "asl", "A")]
                elif sql.startswith("SELECT quiz_id FROM quizzes"):
                    self.rows = [(7,), (8,)]
                elif sql.startswith("SELECT user_id, quiz_id, score"):
                    self.rows = [(1, 7, 2, 10), (1, 7, 8, 10), (2, 7, 6, 10)]
                else:
                    raise AssertionError(sql)

            def fetchall(self):
                return self.rows

            def close(self):
                pass

        class RankConnection:
            def cursor(self):
                return RankCursor()

            def close(self):
                pass

        with patch("user.learning.get_db_connection", return_value=RankConnection()), \
             patch("user.learning.ensure_learning_progress_table"):
            response = self.client.get("/api/learning/rank")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["rank"], 1)
        self.assertEqual(response.get_json()["quiz_percent"], 40.0)


if __name__ == "__main__":
    unittest.main()
