import os
import threading
import time
from pathlib import Path
import mysql.connector
from mysql.connector import Error
from config.environment import load_environment

BASE_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CURRENT_DIR = Path(__file__).resolve().parent

load_environment()

# A fresh connection to the hosted database costs roughly 600 ms (TCP + TLS +
# auth), and every admin request opened two or three of them, so pages spent
# seconds just connecting. Closed connections are kept here and handed out
# again instead: the public get_db_connection()/close() contract is unchanged,
# close() simply returns the session to this pool.
MAX_IDLE_CONNECTIONS = max(1, int(os.getenv("DB_POOL_SIZE", "4")))
# A session parked longer than this must prove it can still run a query before
# it is handed out: the hosted TiDB drops idle sockets and ping() alone did not
# always notice, which surfaced as a one-off 503 on the next page load.
IDLE_REVALIDATE_SECONDS = 30

_idle_connections = []
_idle_lock = threading.Lock()


def _resolve_ssl_ca_path():
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


def _connection_config():
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

    return conn_config


def _connect():
    try:
        return mysql.connector.connect(**_connection_config())
    except Error as err:
        print(f"[Database Connection Error]: {err}")
        raise


def _discard(connection):
    try:
        connection.close()
    except Exception:  # noqa: BLE001 - already gone
        pass


def _acquire():
    with _idle_lock:
        idle_for = 0.0
        connection = None
        if _idle_connections:
            connection, released_at = _idle_connections.pop()
            idle_for = time.monotonic() - released_at
    if connection is None:
        return _connect()

    # is_connected() is a round trip on this connector, so liveness is proven
    # by the ROLLBACK below (and by a ping only for long-idle sessions) instead
    # of on every checkout.
    try:
        if getattr(connection, "in_transaction", False):
            connection.rollback()
        if idle_for > IDLE_REVALIDATE_SECONDS:
            connection.ping(reconnect=True, attempts=1, delay=0)
            probe = connection.cursor()
            try:
                probe.execute("SELECT 1")
                probe.fetchall()
            finally:
                probe.close()
        return connection
    except Exception:  # noqa: BLE001 - the session died while it was idle
        _discard(connection)
        return _connect()


def _release(connection):
    # Never park a session with a transaction, its snapshot or its locks open;
    # ROLLBACK doubles as the liveness check (success proves the session works).
    try:
        connection.rollback()
    except Exception:  # noqa: BLE001 - broken session, never reuse it
        _discard(connection)
        return

    with _idle_lock:
        if len(_idle_connections) < MAX_IDLE_CONNECTIONS:
            _idle_connections.append((connection, time.monotonic()))
            return
    _discard(connection)


class ReusableConnection:
    """Proxy that behaves like the raw connection, except close() recycles it."""

    __slots__ = ("_connection",)

    def __init__(self, connection):
        object.__setattr__(self, "_connection", connection)

    def __getattr__(self, name):
        connection = object.__getattribute__(self, "_connection")
        if connection is None:
            raise AttributeError(name)
        return getattr(connection, name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_connection"), name, value)

    def cursor(self, *args, **kwargs):
        connection = object.__getattribute__(self, "_connection")
        if connection is None:
            raise Error("Connection is closed")
        return connection.cursor(*args, **kwargs)

    def commit(self):
        connection = object.__getattribute__(self, "_connection")
        if connection is not None:
            connection.commit()

    def rollback(self):
        connection = object.__getattribute__(self, "_connection")
        if connection is not None:
            connection.rollback()

    def is_connected(self):
        connection = object.__getattribute__(self, "_connection")
        if connection is None:
            return False
        try:
            return connection.is_connected()
        except Exception:  # noqa: BLE001 - pending result set on a live socket
            return True

    def close(self):
        connection = object.__getattribute__(self, "_connection")
        if connection is None:
            return
        object.__setattr__(self, "_connection", None)
        _release(connection)


def get_db_connection():
    """Check out a MySQL/TiDB connection; close() returns it for reuse."""
    return ReusableConnection(_acquire())


def test_db_connection():
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
        if conn:
            conn.close()


if __name__ == "__main__":
    test_db_connection()
