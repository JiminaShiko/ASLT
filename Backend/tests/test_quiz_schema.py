import unittest

from quiz_schema import QUESTION_COLUMNS, apply_quiz_schema
from admin.quiz_router import ApiError, _validate_managed_quiz


class Cursor:
    def __init__(self, existing):
        self.existing = existing
        self.statements = []
        self.closed = False

    def execute(self, statement, params=None):
        self.statements.append(" ".join(statement.split()))

    def fetchall(self):
        return [(column,) for column in self.existing]

    def close(self):
        self.closed = True


class Connection:
    def __init__(self, existing):
        self.cur = Cursor(existing)
        self.committed = False

    def cursor(self):
        return self.cur

    def commit(self):
        self.committed = True


class QuizSchemaTests(unittest.TestCase):
    def test_existing_columns_are_preserved_and_missing_columns_added(self):
        conn = Connection(["question_id", "question_kind"])
        apply_quiz_schema(conn)
        alters = [sql for sql in conn.cur.statements if sql.startswith("ALTER TABLE")]
        self.assertEqual(len(alters), len(QUESTION_COLUMNS) - 1)
        self.assertFalse(any("DROP" in sql for sql in conn.cur.statements))
        self.assertTrue(conn.committed)
        self.assertTrue(conn.cur.closed)

    def test_publish_requires_ten_questions(self):
        with self.assertRaises(ApiError):
            _validate_managed_quiz({"title": "Test", "status": "active", "questions": []})

    def test_multiple_choice_accepts_mixed_media_options(self):
        question = {"kind": "multiple_choice", "text": "Choose a sign", "media_type": "text",
                    "options": [{"type": "text", "value": "hello"},
                                {"type": "image", "value": "https://example.com/sign.png"},
                                {"type": "video", "value": "https://example.com/sign.mp4"},
                                {"type": "text", "value": "bye"}], "correct": 1}
        quiz = _validate_managed_quiz({"title": "Practice", "questions": [question]})
        self.assertEqual(quiz["questions"][0]["correct"], 1)
        self.assertEqual(quiz["questions"][0]["options"][2]["type"], "video")

    def test_missing_source_table_stops_migration(self):
        conn = Connection([])
        with self.assertRaises(RuntimeError):
            apply_quiz_schema(conn)
        self.assertFalse(conn.committed)
        self.assertTrue(conn.cur.closed)


if __name__ == "__main__":
    unittest.main()
