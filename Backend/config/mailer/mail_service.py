import os
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from pathlib import Path
from config.environment import load_environment

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "Front-end"

load_environment()


def _get_required_env(key: str) -> str:
    """Read a required environment variable or raise."""
    val = os.getenv(key, "").strip().strip('"').strip("'")
    if not val:
        raise ValueError(f"Missing required environment variable: {key}")
    return val


def _resolve_paper_image_path() -> Path | None:
    """Locate the MAIL_PAPER_BG image, if configured."""
    configured = os.getenv("MAIL_PAPER_BG", "").strip().strip('"').strip("'")
    if not configured:
        return None

    candidates = [
        Path(configured),
        PROJECT_ROOT / configured,
        FRONTEND_DIR / configured,
        BACKEND_DIR / configured,
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def _build_paper_otp_html(
    recipient_name: str,
    otp_code: str,
    otp_type_label: str,
    purpose_text: str,
) -> str:
    """Botanical OTP email; % padding + media queries keep text inside the paper strip."""
    safe_name = recipient_name.strip() if recipient_name and recipient_name.strip() else "Friend"
    serif_stack = "Palatino, Georgia, Garamond, serif"

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Signrest — {otp_type_label}</title>
  <style type="text/css">
    @media only screen and (max-width: 520px) {{
      .paper-cell {{
        padding: 33% 8% 27% 37% !important;
      }}
      .otp-welcome {{
        font-size: 11px !important;
        margin-bottom: 2px !important;
      }}
      .otp-desc {{
        font-size: 9.5px !important;
        line-height: 1.2 !important;
        margin-bottom: 4px !important;
      }}
      .otp-box-wrap {{
        padding: 1px 9px !important;
        margin: 2px 0 4px 0 !important;
      }}
      .otp-digits {{
        font-size: 13.5px !important;
        letter-spacing: 3px !important;
      }}
      .otp-signoff {{
        font-size: 9px !important;
        line-height: 1.2 !important;
      }}
    }}
  </style>
</head>
<body style="margin:0; padding:16px 8px; background-color:#ffffff; font-family:{serif_stack}; color:#1e3b24;">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background-color:#ffffff;">
    <tr>
      <td align="center" style="background-color:#ffffff;">
        <!-- Main Torn Paper & Eucalyptus Card Container -->
        <table role="presentation" width="600" cellspacing="0" cellpadding="0" border="0"
               style="max-width:600px; width:100%; background-color:#ffffff; background-image:url('cid:signrest_paper_bg'); background-size:100% auto; background-repeat:no-repeat; background-position:center center;">
          <tr>
            <td class="paper-cell" style="padding: 34.5% 10% 28.5% 38.5%; text-align:left; vertical-align:middle; font-family:{serif_stack};">
              <p class="otp-welcome" style="margin:0 0 3px 0; font-size:14px; line-height:1.25; font-weight:normal; color:#1e3b24; font-family:{serif_stack};">
                Welcome, <span style="font-weight:bold; color:#152e1b; font-family:{serif_stack};">{safe_name}</span>
              </p>

              <p class="otp-desc" style="margin:0 0 5px 0; font-size:12px; line-height:1.25; font-weight:normal; color:#27472e; font-family:{serif_stack};">
                {purpose_text} Your <span style="font-weight:bold; color:#152e1b; font-family:{serif_stack};">{otp_type_label}</span> is:
              </p>

              <div class="otp-box-wrap" style="margin:3px 0 6px 0; display:inline-block; padding:2px 14px; border:1px dashed #42694a; border-radius:6px; background-color:rgba(255,255,255,0.68);">
                <span class="otp-digits" style="font-size:18px; font-weight:bold; letter-spacing:5px; color:#152e1b; font-family:{serif_stack};">
                  {otp_code}
                </span>
              </div>

              <p class="otp-signoff" style="margin:0; font-size:11px; line-height:1.25; font-style:italic; font-weight:normal; color:#2c4d33; font-family:{serif_stack};">
                With warmth, from our <span style="font-weight:bold; color:#152e1b; font-family:{serif_stack};">Signrest</span> community.
              </p>
            </td>
          </tr>
        </table>

        <!-- Subtle footer note -->
        <p style="margin:8px 0 0 0; font-size:11px; color:#526b57; font-family:{serif_stack};">
          This {otp_type_label} expires in 10 minutes. If you didn’t request this, you can safely ignore this email.
        </p>
      </td>
    </tr>
  </table>
</body>
</html>
"""


def send_email(
    to_email: str,
    subject: str,
    html_body: str,
    plain_body: str = "",
    inline_images: dict[str, Path] | None = None,
) -> tuple[bool, str]:
    """Send one SMTP email using host/credential settings from the environment."""
    try:
        smtp_host = _get_required_env("MAIL_SERVER")
        smtp_port = int(_get_required_env("MAIL_PORT"))
        smtp_user = _get_required_env("MAIL_USERNAME")
        smtp_pass = _get_required_env("MAIL_PASSWORD").replace(" ", "")
        sender_name = _get_required_env("MAIL_SENDER_NAME")
        use_tls_env = os.getenv("MAIL_USE_TLS", "").strip().lower()
        use_tls = (use_tls_env == "true") or (smtp_port == 587 and use_tls_env != "false")
    except ValueError as env_err:
        print(f"[Mailer Config Error]: {env_err}")
        return False, str(env_err)

    msg_root = MIMEMultipart("related")
    msg_root["Subject"] = subject
    msg_root["From"] = formataddr((sender_name, smtp_user))
    msg_root["To"] = to_email

    msg_alternative = MIMEMultipart("alternative")
    msg_root.attach(msg_alternative)

    if plain_body:
        msg_alternative.attach(MIMEText(plain_body, "plain", "utf-8"))
    msg_alternative.attach(MIMEText(html_body, "html", "utf-8"))

    if inline_images:
        for cid_name, img_path in inline_images.items():
            if img_path and img_path.is_file():
                with open(img_path, "rb") as f:
                    img_part = MIMEImage(f.read())
                    img_part.add_header("Content-ID", f"<{cid_name}>")
                    img_part.add_header("Content-Disposition", "inline", filename=img_path.name)
                    msg_root.attach(img_part)

    try:
        if smtp_port == 465 and not use_tls:
            server = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=20)
        else:
            server = smtplib.SMTP(smtp_host, smtp_port, timeout=20)
            server.ehlo()
            if use_tls:
                server.starttls()
                server.ehlo()

        with server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, [to_email], msg_root.as_string())

        print(f"[Mailer Success]: OTP email sent to {to_email}")
        return True, "Email sent successfully."
    except Exception as err:
        print(f"[Mailer Error] ({smtp_host}:{smtp_port} as {smtp_user}): {err}")
        return False, f"Failed to send email: {err}"


def send_otp_email(
    to_email: str,
    otp_code: str,
    purpose: str = "signup",
    recipient_name: str | None = None,
) -> tuple[bool, str]:
    """Send the styled OTP email; purpose is signup, resend, reset_password, or login."""
    display_name = recipient_name or to_email.split("@")[0]
    normalized_purpose = (purpose or "signup").strip().lower()

    if normalized_purpose == "resend":
        otp_type_label = "Resent OTP"
        subject = "Signrest — Your Resent OTP Code"
        purpose_text = "Here is your newly requested code."
    elif normalized_purpose in ("reset_password", "forgot_password"):
        otp_type_label = "Password Reset OTP"
        subject = "Signrest — Password Reset OTP"
        purpose_text = "We received a request to reset your password."
    elif normalized_purpose == "login":
        otp_type_label = "Login OTP"
        subject = "Signrest — Login Verification OTP"
        purpose_text = "Please verify your sign-in attempt."
    else:
        otp_type_label = "Sign Up OTP"
        subject = "Welcome to Signrest — Your Sign Up OTP"
        purpose_text = "Thank you for joining us."

    plain_body = (
        f"Welcome, {display_name}!\n"
        f"{purpose_text} Your {otp_type_label} is: {otp_code}\n"
        f"With warmth, from our Signrest community."
    )

    html_body = _build_paper_otp_html(
        recipient_name=display_name,
        otp_code=otp_code,
        otp_type_label=otp_type_label,
        purpose_text=purpose_text,
    )

    paper_path = _resolve_paper_image_path()
    inline_images = {"signrest_paper_bg": paper_path} if paper_path else None

    return send_email(
        to_email=to_email,
        subject=subject,
        html_body=html_body,
        plain_body=plain_body,
        inline_images=inline_images,
    )