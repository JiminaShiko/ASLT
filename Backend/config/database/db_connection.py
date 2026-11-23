import os
from pathlib import Path
import mysql.connector
from mysql.connector import Error
from config.environment import load_environment

BASE_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CURRENT_DIR = Path(__file__).resolve().parent

load_environment()


def _resolve_ssl_ca_path() -> str | None:
    """Find DB_SSL_CA across localhost and Render Secret Files locations."""
    ssl_ca = os.getenv("DB_SSL_CA", "").strip()
    if not ssl_ca:
        return None

    candidate_paths = [
        Path(ssl_ca),  # absolute or relative to CWD
        BASE_DIR / ssl_ca,
        PROJECT_ROOT / ssl_ca,
        CURRENT_DIR / ssl_ca,
        Path("/etc/secrets") / Path(ssl_ca).name,  # Render Secret Files default location
    ]

    for path in candidate_paths:
        if path.is_file():
            return str(path.resolve())

    return None


def get_db_connection():
    """Open a new MySQL/TiDB connection from environment variables."""
    db_host = os.getenv("DB_HOST")
    db_port = int(os.getenv("DB_PORT", "4000"))
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASS")
    db_name = os.getenv("DB_NAME")
    ssl_ca_path = _resolve_ssl_ca_path()

    if not all([db_host, db_user, db_pass, db_name]):
        raise ValueError(
            "Missing required database environment variables: DB_HOST, DB_USER, DB_PASS, or DB_NAME."
        )

    conn_config = {
        "host": db_host,
        "port": db_port,
        "user": db_user,
        "password": db_pass,
        "database": db_name,
        "autocommit": False,
        "connection_timeout": 10,
    }

    if ssl_ca_path:
        conn_config["ssl_ca"] = ssl_ca_path
        conn_config["ssl_disabled"] = False
        # Aiven/TiDB fail with cert verification enabled in mysql-connector-python.
        verify_ssl = os.getenv("DB_SSL_VERIFY", "false").lower() == "true"
        conn_config["ssl_verify_cert"] = verify_ssl
        conn_config["ssl_verify_identity"] = False

    try:
        connection = mysql.connector.connect(**conn_config)
        return connection
    except Error as err:
        print(f"[Database Connection Error]: {err}")
        raise


def test_db_connection() -> bool:
    """Verify database connectivity."""
    conn = None
    try:
        conn = get_db_connection()
        if conn and conn.is_connected():
            print("[Database]: Connected successfully.")
            return True
        return False
    except Exception as e:
        print(f"[Database]: Connection check failed -> {e}")
        return False
    finally:
        if conn and conn.is_connected():
            conn.close()


if __name__ == "__main__":
    test_db_connection()