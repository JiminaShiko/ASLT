import os
from pathlib import Path
import mysql.connector
from mysql.connector import Error
from config.environment import load_environment

# Resolve project root directory (root/config/database/db_connection.py -> root/)
BASE_DIR = Path(__file__).resolve().parents[2]
CURRENT_DIR = Path(__file__).resolve().parent

load_environment()


def _resolve_ssl_ca_path() -> str | None:
    """
    Resolves DB_SSL_CA path cleanly across localhost, config/database/, and Render Secret Files.
    """
    ssl_ca = os.getenv("DB_SSL_CA", "").strip()
    if not ssl_ca:
        return None

    candidate_paths = [
        Path(ssl_ca),                         # Absolute path or relative to CWD
        BASE_DIR / ssl_ca,                    # Relative to project root (e.g., ASLT/ca.pem)
        CURRENT_DIR / ssl_ca,                 # Relative to config/database/ca.pem
        Path("/etc/secrets") / Path(ssl_ca).name,  # Render Secret Files default location
    ]

    for path in candidate_paths:
        if path.is_file():
            return str(path.resolve())

    return None


def get_db_connection():
    """
    Creates and returns a new MySQL/TiDB database connection using environment variables.
    Works seamlessly on both localhost (.env) and Render (Environment Variables).
    """
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
        # Cloud MySQL providers (Aiven/TiDB) often require ssl_verify_cert=False with mysql-connector-python
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
    """Quick helper to verify database connectivity."""
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