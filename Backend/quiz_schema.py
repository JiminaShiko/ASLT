"""Additive schema preparation for mixed-media quizzes.

Run explicitly after reviewing and backing up the database. Existing quiz and
admin-question data is never deleted by this migration.
"""

from config.database.db_connection import get_db_connection


QUESTION_COLUMNS = {
    "question_kind": "VARCHAR(24) NOT NULL DEFAULT 'multiple_choice'",
    "prompt_media_type": "VARCHAR(10) NOT NULL DEFAULT 'text'",
    "target_category": "VARCHAR(20) NULL",
    "target_item_key": "VARCHAR(100) NULL",
    "display_order": "INT NOT NULL DEFAULT 0",
}


def apply_quiz_schema(connection):
    cur = connection.cursor()
    try:
        cur.execute(
            "SELECT COLUMN_NAME FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = %s",
            ("quiz_questions",),
        )
        existing = {row[0].lower() for row in cur.fetchall()}
        if not existing:
            raise RuntimeError("quiz_questions table is missing; migration stopped.")
        for name, definition in QUESTION_COLUMNS.items():
            if name not in existing:
                # Names and definitions come from the fixed mapping above, not user input.
                cur.execute(f"ALTER TABLE quiz_questions ADD COLUMN `{name}` {definition}")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS quiz_attempts (
                attempt_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                quiz_id INT NOT NULL,
                user_id INT NOT NULL,
                attempt_number TINYINT NOT NULL,
                status VARCHAR(20) NOT NULL DEFAULT 'in_progress',
                question_snapshot JSON NOT NULL,
                answer_state JSON NOT NULL,
                started_at DATETIME NOT NULL,
                completed_at DATETIME NULL,
                UNIQUE KEY uq_quiz_user_attempt (quiz_id, user_id, attempt_number),
                INDEX idx_quiz_attempt_user_status (user_id, quiz_id, status)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        connection.commit()
    finally:
        cur.close()


if __name__ == "__main__":
    conn = get_db_connection()
    try:
        apply_quiz_schema(conn)
        print("Quiz schema prepared. Existing quiz and admin tables were preserved.")
    finally:
        conn.close()
