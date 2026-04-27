import hmac
import logging
import re
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from app.config import settings

logger = logging.getLogger(__name__)

_INJECTION_PATTERNS = re.compile(
    r"ignore previous instructions|forget everything|you are now|"
    r"<script|</script|javascript:|onerror=|onload=",
    re.IGNORECASE,
)

_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

MAX_COLUMNS = 50
MAX_ROW_COUNT = 10_000
MIN_ROW_COUNT = 1


def sanitize_text(value: str) -> str:
    cleaned = _HTML_TAG_PATTERN.sub("", value)
    cleaned = cleaned.replace("\r", "").replace("\n", " ")
    return cleaned.strip()


def contains_injection(value: str) -> bool:
    return bool(_INJECTION_PATTERNS.search(value))


def sanitize_log_field(value: str) -> str:
    return value.replace("\r", "").replace("\n", "")


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def create_access_token(subject: str, expires_delta: timedelta | None = None) -> str:
    now = datetime.now(timezone.utc)
    expire = now + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload: dict = {"sub": subject, "exp": expire, "iat": now}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_access_token(token: str) -> dict:
    from jwt import ExpiredSignatureError, InvalidTokenError

    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
            options={"require": ["exp", "sub"]},
        )
        return payload
    except ExpiredSignatureError as err:
        raise ValueError("Token has expired") from err
    except InvalidTokenError as err:
        raise ValueError("Invalid token") from err


def secure_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


def validate_row_count(row_count: int) -> int:
    if row_count < MIN_ROW_COUNT:
        raise ValueError(f"row_count must be at least {MIN_ROW_COUNT}")
    if row_count > MAX_ROW_COUNT:
        raise ValueError(f"row_count must not exceed {MAX_ROW_COUNT}")
    return row_count


def validate_column_count(columns: list) -> list:
    if len(columns) > MAX_COLUMNS:
        raise ValueError(f"Schema must not exceed {MAX_COLUMNS} columns")
    return columns


def validate_domain(domain: str) -> str:
    allowed = {"healthcare", "finance", "retail", "hr", "iot", "custom"}
    if domain not in allowed:
        raise ValueError(f"domain must be one of: {', '.join(sorted(allowed))}")
    return domain
