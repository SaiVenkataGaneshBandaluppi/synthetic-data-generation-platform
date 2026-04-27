import json
import logging

from app.config import settings

logger = logging.getLogger(__name__)

_client = None


def _build_client():
    global _client
    if _client is not None:
        return _client
    if not settings.GROQ_API_KEY:
        return None
    try:
        from groq import Groq

        _client = Groq(api_key=settings.GROQ_API_KEY)
        return _client
    except Exception as err:
        logger.warning("Failed to build Groq client: %s", err)
        return None


def groq_complete(
    prompt: str,
    system: str,
    max_tokens: int = 2048,
    model: str = "llama-3.3-70b-versatile",
) -> dict | None:
    client = _build_client()
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            timeout=10.0,
            response_format={"type": "json_object"},
        )
        if not response.choices:
            logger.warning("Empty choices from Groq response")
            return None
        raw = response.choices[0].message.content
        if not raw:
            return None
        return json.loads(raw)
    except Exception as err:
        logger.warning("Groq completion failed: %s", err)
        return None


def reset_client() -> None:
    global _client
    _client = None
