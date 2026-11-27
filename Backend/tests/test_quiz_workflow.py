"""Exercise the admin-to-learner multiple-choice flow without a live database."""
import sqlite3
import io
import unittest
from unittest.mock import patch

from flask import Flask, g

from admin.quiz_router import admin_quiz_bp
from user.learning import learning_bp
from PIL import Image


class Cursor:
    def __init__(self, connection, dictionary=False):
        self.cursor = connection.cursor()
        self.dictionary = dictionary

    def execute(self, sql, params=()):
        sql = sql.replace("%s", "?").replace(" FOR UPDATE", "").replace("UTC_TIMESTAMP()", "CURRENT_TIMESTAMP")
        self.cursor.execute(sql, params)

    @property
    def lastrowid(self):
        return self.cursor.lastrowid

    def _convert(self, row):
        if row is None:
            return None
        return dict(row) if self.dictionary else tuple(row)

    def fetchone(self):
        return self._convert(self.cursor.fetchone())

    def fetchall(self):
        return [self._convert(row) for row in self.cursor.fetchall()]

    def close(self):
        self.cursor.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class Connection:
    def __init__(self):
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE users (user_id INTEGER PRIMARY KEY, role TEXT, status TEXT);
            INSERT INTO users VALUES (1,'user','active'), (2,'user','active'), (99,'admin','active');
            CREATE TABLE quizzes (quiz_id INTEGER PRIMARY KEY, created_by INTEGER, title TEXT, description TEXT, status TEXT, created_at TEXT);
            CREATE TABLE quiz_questions (question_id INTEGER PRIMARY KEY, quiz_id INTEGER, question_text TEXT, image_path TEXT, correct_answer TEXT, options TEXT, question_kind TEXT, prompt_media_type TEXT, target_category TEXT, target_item_key TEXT, display_order INTEGER);
            CREATE TABLE quiz_results (result_id INTEGER PRIMARY KEY, quiz_id INTEGER, user_id INTEGER, score INTEGER, total_questions INTEGER, completed_at TEXT);
            CREATE TABLE user_learning_progress (user_id INTEGER, category TEXT, item_key TEXT);
            CREATE TABLE quiz_attempts (attempt_id INTEGER PRIMARY KEY, quiz_id INTEGER, user_id INTEGER, attempt_number INTEGER, status TEXT, question_snapshot TEXT, answer_state TEXT, started_at TEXT, completed_at TEXT, UNIQUE(quiz_id,user_id,attempt_number));
        """)

    def cursor(self, dictionary=False):
        return Cursor(self.db, dictionary)

    def commit(self):
        self.db.commit()

    def rollback(self):
        self.db.rollback()

    def close(self):
        pass


class QuizWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.connection = Connection()
        self.patches = [patch("admin.quiz_router.get_db_connection", return_value=self.connection),
                        patch("user.learning.get_db_connection", return_value=self.connection)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        app = Flask(__name__)
        app.secret_key = "test"
        app.register_blueprint(admin_quiz_bp)
        app.register_blueprint(learning_bp)

        @app.before_request
        def account():
            g.signrest_account = {"user_id": 99, "role": "admin", "status": "active"}

        self.client = app.test_client()

    def sign_in(self, user_id, role):
        with self.client.session_transaction() as sess:
            sess["user_id"] = user_id
            sess["user_email"] = "test@example.test"
            sess["user_role"] = role

    def create_quiz(self):
        self.sign_in(99, "admin")
        question = {"kind": "multiple_choice", "text": "Choose the answer", "media_type": "text",
                    "options": [{"type": "text", "value": value} for value in ["A", "B", "C", "D"]],
                    "correct": 0}
        response = self.client.post("/api/admin/quiz/managed-quizzes", headers={"X-Signrest-Request": "admin"},
                                    json={"title": "Test", "status": "active", "questions": [question] * 10})
        self.assertEqual(response.status_code, 201, response.get_json())
        return response.get_json()["quiz_id"]

    def test_publish_resume_finish_and_three_attempt_limit(self):
        quiz_id = self.create_quiz()
        self.sign_in(1, "user")
        self.assertEqual(self.client.get(f"/api/quizzes/{quiz_id}").status_code, 409)
        for number in range(1, 4):
            response = self.client.post(f"/api/quizzes/{quiz_id}/attempts")
            self.assertEqual(response.status_code, 201)
            attempt = response.get_json()
            self.assertEqual(attempt["attempt_number"], number)
            self.assertNotIn("correct_answer", attempt["questions"][0])
            self.assertEqual(self.client.post(f"/api/quiz-attempts/{attempt['attempt_id']}/finish").status_code, 400)
            first = attempt["questions"][0]["question_id"]
            result = self.client.post(f"/api/quiz-attempts/{attempt['attempt_id']}/answer", json={"question_id": first, "answer": "0"})
            self.assertEqual(result.status_code, 200)
            resumed = self.client.post(f"/api/quizzes/{quiz_id}/attempts").get_json()
            self.assertEqual(resumed["attempt_id"], attempt["attempt_id"])
            self.assertEqual(resumed["answers"][str(first)], "0")
            if number == 1:
                listing = self.client.get("/api/quizzes").get_json()["quizzes"][0]
                self.assertEqual(listing["progress"], {"answered": 1, "total": 10})
                self.sign_in(2, "user")
                denied = self.client.post(f"/api/quiz-attempts/{attempt['attempt_id']}/answer", json={"question_id": first, "answer": "0"})
                self.assertEqual(denied.status_code, 404)
                self.sign_in(1, "user")
            for question in attempt["questions"][1:]:
                result = self.client.post(f"/api/quiz-attempts/{attempt['attempt_id']}/answer", json={"question_id": question["question_id"], "answer": "0"})
                self.assertEqual(result.status_code, 200)
            result = self.client.post(f"/api/quiz-attempts/{attempt['attempt_id']}/finish")
            self.assertEqual(result.get_json()["score"], 10)
            self.assertEqual(self.client.post(f"/api/quiz-attempts/{attempt['attempt_id']}/finish").status_code, 200)
        self.assertEqual(self.client.post(f"/api/quizzes/{quiz_id}/attempts").status_code, 409)
        self.assertEqual(self.connection.db.execute("SELECT COUNT(*) FROM quiz_results").fetchone()[0], 3)
        listing = self.client.get("/api/quizzes").get_json()["quizzes"][0]
        self.assertEqual(len(listing["results"]), 3)
        self.assertEqual(listing["best_result"]["best_score"], 10)
        with patch("user.learning.ensure_learning_progress_table"):
            rank = self.client.get("/api/learning/rank")
        self.assertEqual(rank.status_code, 200)
        self.assertEqual(rank.get_json()["quiz_percent"], 100.0)

    def test_media_upload_is_limited_to_admin_and_valid_files(self):
        self.sign_in(99, "admin")
        content = io.BytesIO()
        Image.new("RGB", (2, 2), "green").save(content, format="PNG")
        content.seek(0)
        with patch("admin.quiz_router.upload_to_cloudinary", return_value="https://example.test/quiz.png") as upload:
            response = self.client.post("/api/admin/quiz/media", headers={"X-Signrest-Request": "admin"},
                                        data={"file": (content, "question.png")})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_json()["type"], "image")
        upload.assert_called_once()
        invalid = self.client.post("/api/admin/quiz/media", headers={"X-Signrest-Request": "admin"},
                                   data={"file": (io.BytesIO(b"not an image"), "bad.png")})
        self.assertEqual(invalid.status_code, 422)
        self.sign_in(1, "user")
        denied = self.client.post("/api/admin/quiz/media", headers={"X-Signrest-Request": "admin"},
                                 data={"file": (io.BytesIO(b"fake"), "image.png")})
        self.assertEqual(denied.status_code, 403)

    def test_admin_can_edit_and_publish_draft_but_not_change_started_quiz(self):
        self.sign_in(99, "admin")
        headers = {"X-Signrest-Request": "admin"}
        created = self.client.post("/api/admin/quiz/managed-quizzes", headers=headers,
                                   json={"title": "Draft", "status": "draft", "questions": []})
        self.assertEqual(created.status_code, 201)
        quiz_id = created.get_json()["quiz_id"]
        question = {"kind": "multiple_choice", "text": "Choose a sign", "media_type": "image",
                    "media_url": "https://example.test/prompt.png", "correct": 1,
                    "options": [{"type": "text", "value": "A"},
                                {"type": "video", "value": "https://example.test/sign.mp4"},
                                {"type": "image", "value": "https://example.test/sign.png"},
                                {"type": "text", "value": "D"}]}
        data = {"title": "Published", "status": "active", "questions": [question] * 10}
        response = self.client.put(f"/api/admin/quiz/managed-quizzes/{quiz_id}", headers=headers, json=data)
        self.assertEqual(response.status_code, 200, response.get_json())
        details = self.client.get(f"/api/admin/quiz/managed-quizzes/{quiz_id}").get_json()["item"]
        self.assertEqual(details["status"], "active")
        self.assertEqual(details["questions"][0]["options"][1]["type"], "video")
        self.sign_in(1, "user")
        started = self.client.post(f"/api/quizzes/{quiz_id}/attempts")
        self.assertEqual(started.status_code, 201)
        self.sign_in(99, "admin")
        blocked = self.client.put(f"/api/admin/quiz/managed-quizzes/{quiz_id}", headers=headers, json=data)
        self.assertEqual(blocked.status_code, 409)

    def test_publishing_requires_ten_questions_and_disallows_camera_for_now(self):
        self.sign_in(99, "admin")
        choice = {"kind": "multiple_choice", "text": "Choose", "media_type": "text",
                  "options": [{"type": "text", "value": value} for value in ["A", "B", "C", "D"]], "correct": 0}
        headers = {"X-Signrest-Request": "admin"}
        short = self.client.post("/api/admin/quiz/managed-quizzes", headers=headers,
                                 json={"title": "Short", "status": "active", "questions": [choice] * 9})
        self.assertEqual(short.status_code, 422)
        camera = {"kind": "camera", "text": "Sign A", "media_type": "text", "target_category": "asl", "target_item_key": "A"}
        mixed = self.client.post("/api/admin/quiz/managed-quizzes", headers=headers,
                                 json={"title": "Mixed", "status": "active", "questions": [choice] * 9 + [camera]})
        self.assertEqual(mixed.status_code, 422)


if __name__ == "__main__":
    unittest.main()
