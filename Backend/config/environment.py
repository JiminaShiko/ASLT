from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent


def _find_env_file() -> Path | None:
    for directory in (PROJECT_ROOT, BACKEND_DIR):
        env_file = directory / ".env"
        if env_file.is_file():
            return env_file
        candidates = [path for path in directory.glob("*.env") if path.is_file()]
        if len(candidates) > 1:
            raise RuntimeError(
                "Multiple environment files found. Use a single root-level .env file."
            )
        if candidates:
            return candidates[0]
    return None


def load_environment() -> None:
    env_file = _find_env_file()
    if env_file is not None:
        load_dotenv(env_file, override=False)
