import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config.cloudinary import upload_to_cloudinary
from config.database.db_connection import get_db_connection


def upload_and_register_effect(
    file_path: str,
    effect_name: str,
    effect_type: str = "both",  # 'trail', 'click', or 'both'
    status: str = "active",
):
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Asset not found: {path}")

    if effect_type not in {"trail", "click", "both"}:
        raise ValueError("effect_type must be 'trail', 'click', or 'both'")

    print(f"Uploading '{effect_name}' ({path.name}) to Cloudinary...")
    secure_url = upload_to_cloudinary(
        file_source=path,
        folder="cursor_effects",  # uploads into `signrest/cursor_effects`
        resource_type="image",
    )

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO cursor_effects (effect_name, effect_path, effect_type, status, created_at)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (effect_name, secure_url, effect_type, status, datetime.now(timezone.utc)),
    )
    conn.commit()
    new_id = cur.lastrowid
    cur.close()
    conn.close()

    print(f"Saved -> effect_id={new_id}, type={effect_type}, url={secure_url}")
    return new_id, secure_url


if __name__ == "__main__":
    # Example usage: replace with your local bg-removed image/GIF paths and run `python Backend/upload_cursor_effects.py`
    sample_assets = [
        # ("../Front-end/assets/images/effects/forest_leaf.png", "Forest Leaf", "both"),
        # ("../Front-end/assets/images/effects/sage_sparkle.png", "Sage Sparkle", "trail"),
        # ("../Front-end/assets/images/effects/bloom_ring.png", "Bloom Burst", "click"),
    ]

    if not sample_assets:
        path = input("Local image/GIF path: ").strip().strip('"')
        name = input("Effect name: ").strip()
        etype = input("Effect type (trail / click / both) [both]: ").strip().lower() or "both"
        upload_and_register_effect(path, name, etype)
    else:
        for asset_path, name, etype in sample_assets:
            upload_and_register_effect(asset_path, name, etype)