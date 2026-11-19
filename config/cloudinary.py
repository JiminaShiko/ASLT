import hashlib
import os
import time
import urllib.parse
import urllib.request
import json
import uuid
from pathlib import Path
from config.environment import load_environment

BASE_DIR = Path(__file__).resolve().parents[1]
load_environment()


def _get_cloudinary_config() -> tuple[str, str, str]:
    """
    Reads CLOUDINARY_URL (cloudinary://api_key:api_secret@cloud_name)
    or individual CLOUDINARY_* env vars for both Localhost and Render.
    """
    cloudinary_url = os.getenv("CLOUDINARY_URL", "").strip()
    if cloudinary_url:
        parsed = urllib.parse.urlparse(cloudinary_url)
        cloud_name = parsed.hostname or ""
        api_key = urllib.parse.unquote(parsed.username or "")
        api_secret = urllib.parse.unquote(parsed.password or "")
        if cloud_name and api_key and api_secret:
            return cloud_name, api_key, api_secret

    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME", "").strip()
    api_key = os.getenv("CLOUDINARY_API_KEY", "").strip()
    api_secret = os.getenv("CLOUDINARY_API_SECRET", "").strip()

    if not cloud_name or not api_key or not api_secret:
        raise RuntimeError(
            "Cloudinary credentials are missing. Please set CLOUDINARY_URL in your .env or Render dashboard."
        )

    return cloud_name, api_key, api_secret


def upload_to_cloudinary(
    file_source,
    folder: str = "general",
    filename: str = "upload.png",
    resource_type: str = "auto",
) -> str:
    """
    Reusable helper across Signrest for user/admin photos & files.
    Accepts either a file path (str/Path), raw bytes, or a file-like stream (.read()).
    Uploads to `signrest/<folder>` on Cloudinary and returns the public HTTPS `secure_url`.
    """
    cloud_name, api_key, api_secret = _get_cloudinary_config()

    if isinstance(file_source, (str, Path)):
        path_obj = Path(file_source)
        if not path_obj.is_file():
            raise FileNotFoundError(f"File does not exist: {file_source}")
        file_bytes = path_obj.read_bytes()
        filename = path_obj.name
    elif isinstance(file_source, bytes):
        file_bytes = file_source
    elif hasattr(file_source, "read"):
        file_bytes = file_source.read()
    else:
        raise ValueError("Unsupported file_source passed to upload_to_cloudinary.")

    timestamp = str(int(time.time()))
    target_folder = f"signrest/{folder.strip('/')}"

    # Cloudinary SHA-1 signature over sorted non-empty parameters
    params_to_sign = {
        "folder": target_folder,
        "timestamp": timestamp,
    }
    sign_string = "&".join(f"{k}={v}" for k, v in sorted(params_to_sign.items())) + api_secret
    signature = hashlib.sha1(sign_string.encode("utf-8")).hexdigest()

    boundary = f"----SignrestBoundary{uuid.uuid4().hex}"
    fields = {
        "api_key": api_key,
        "timestamp": timestamp,
        "folder": target_folder,
        "signature": signature,
    }

    body_parts: list[bytes] = []
    for key, val in fields.items():
        body_parts.append(f"--{boundary}\r\n".encode("utf-8"))
        body_parts.append(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        body_parts.append(f"{val}\r\n".encode("utf-8"))

    body_parts.append(f"--{boundary}\r\n".encode("utf-8"))
    body_parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8")
    )
    body_parts.append(b"Content-Type: application/octet-stream\r\n\r\n")
    body_parts.append(file_bytes)
    body_parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    payload = b"".join(body_parts)
    upload_url = f"https://api.cloudinary.com/v1_1/{cloud_name}/{resource_type}/upload"

    req = urllib.request.Request(
        upload_url,
        data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    secure_url = result.get("secure_url")
    if not secure_url:
        raise RuntimeError(f"Cloudinary upload failed: {result}")

    return secure_url