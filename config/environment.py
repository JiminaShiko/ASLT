from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent


def load_environment() -> None:
    env_file = BASE_DIR / ".env"
    if not env_file.is_file():
        candidates = [path for path in BASE_DIR.glob("*.env") if path.is_file()]
        if len(candidates) > 1:
            raise RuntimeError(
                "Multiple environment files found. Use a single root-level .env file."
            )
        if candidates:
            env_file = candidates[0]

    load_dotenv(env_file, override=False)
