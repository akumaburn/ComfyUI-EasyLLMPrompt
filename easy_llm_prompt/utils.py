"""Utility functions: HTTP helpers, URL normalization, error handling."""

import json
import logging
import re
import socket
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


class LLMResponseError(Exception):
    """Raised when the LLM returns an unexpected or unusable response."""


def post_json_request(url, payload, timeout=60, headers=None):
    """POST a JSON payload to *url* and return the parsed JSON response.

    Raises ``LLMResponseError`` with a human-readable message on
    any failure (network, HTTP status, timeout, malformed JSON).
    """
    data = json.dumps(payload).encode("utf-8")

    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)

    try:
        req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        if e.code == 401:
            msg = (
                "Authentication failed (401). "
                "If this endpoint requires an API key, pass it in the URL "
                "or configure your server to allow anonymous access."
            )
        elif e.code == 404:
            msg = f"Endpoint not found (404). Verify the URL: {url}"
        elif e.code == 422:
            msg = (
                f"Invalid request (422). The model may not support this "
                f"chat format. Response: {body[:200]}"
            )
        else:
            msg = f"HTTP {e.code}: {body[:200]}"
        raise LLMResponseError(msg)

    except urllib.error.URLError as e:
        reason = str(e.reason)
        if "Name or service not known" in reason or "nodename nor servname" in reason:
            raise LLMResponseError(
                f"Could not resolve hostname. Check the URL: {url}"
            )
        if "Connection refused" in reason:
            raise LLMResponseError(
                f"Connection refused. Is the server running at {url}?"
            )
        raise LLMResponseError(f"Connection error: {reason}")

    except json.JSONDecodeError as e:
        raise LLMResponseError(f"Invalid JSON in response: {e}")

    except socket.timeout:
        raise LLMResponseError(
            f"Request timed out after {timeout}s. "
            "Verify the server is running and reachable."
        )

    except OSError as e:
        raise LLMResponseError(f"Network error: {e}")


def normalize_url(raw_url):
    """Strip trailing slashes from a URL."""
    return raw_url.strip().rstrip("/")


def build_chat_url(base_url):
    """Ensure *base_url* has a ``/v1/chat/completions`` path suffix.

    Handles both bare hosts and urls that already end in ``/v1``.
    """
    url = base_url.rstrip("/")

    # Remove any existing chat completions path so we rebuild cleanly
    url = re.sub(r"/chat/completions/?$", "", url)

    if url.endswith("/v1"):
        return url + "/chat/completions"

    return url + "/v1/chat/completions"


def strip_llm_noise(text):
    """Clean common LLM artifacts from generated text."""
    text = text.strip()

    # Balanced surrounding quotes
    if len(text) > 1 and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1].strip()

    # Common preamble / explanation phrases
    prefixes = [
        "Here is", "Here's", "Sure,", "Sure thing", "Certainly,",
        "Of course,", "Enhanced prompt:", "Prompt:", "Result:",
    ]
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    # Code-fence markers the LLM might wrap around the prompt
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()

    # Inline code markers
    if text.startswith("`") and text.endswith("`"):
        text = text[1:-1].strip()

    return text.strip()
