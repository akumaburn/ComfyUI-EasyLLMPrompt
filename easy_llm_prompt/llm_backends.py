"""LLM backend implementations for different providers.

Each backend wraps a single ``chat()`` call.  Currently supported:

* **Ollama** – native /api/chat protocol.
* **OpenAI-compatible** – /v1/chat/completions (also covers llama.cpp
  server, vLLM, LM Studio, LocalAI, etc.).
* **llama.cpp** – same OpenAI-compatible protocol with different
  default URL.
"""

import logging

from .utils import (
    LLMResponseError,
    build_chat_url,
    normalize_url,
    post_json_request,
    strip_llm_noise,
)

logger = logging.getLogger(__name__)


class LLMBackend:
    """Abstract interface for an LLM chat backend."""

    def chat(self, system_prompt, user_message, temperature=0.7,
             max_tokens=512, seed=None):
        """Send a chat request and return the generated text.

        Raises ``LLMResponseError`` on any failure.
        """
        raise NotImplementedError


class OllamaBackend(LLMBackend):
    """Backend for the Ollama_ native API.

    .. _Ollama: https://github.com/ollama/ollama
    """

    def __init__(self, base_url, model, timeout=60):
        self.base_url = normalize_url(base_url)
        self.model = model
        self.timeout = timeout

    def chat(self, system_prompt, user_message, temperature=0.7,
             max_tokens=512, seed=None):
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if seed is not None and seed >= 0:
            payload["options"]["seed"] = seed

        result = post_json_request(url, payload, self.timeout)

        try:
            content = result["message"]["content"]
        except (KeyError, TypeError) as e:
            raise LLMResponseError(
                f"Unexpected Ollama response structure: {e}. "
                f"Response keys: {list(result.keys()) if isinstance(result, dict) else type(result)}"
            )

        return strip_llm_noise(content)


class OpenAIBackend(LLMBackend):
    """Backend for OpenAI-compatible chat endpoints.

    Works with OpenAI_, llama.cpp server, vLLM, LM Studio, LocalAI,
    and any other server that implements the
    ``/v1/chat/completions`` interface.

    .. _OpenAI: https://platform.openai.com/docs/api-reference/chat
    """

    def __init__(self, base_url, model, timeout=60, api_key=None):
        self.base_url = normalize_url(base_url)
        self.model = model
        self.timeout = timeout
        self.api_key = api_key

    def chat(self, system_prompt, user_message, temperature=0.7,
             max_tokens=512, seed=None):
        url = build_chat_url(self.base_url)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if seed is not None and seed >= 0:
            payload["seed"] = seed

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        result = post_json_request(url, payload, self.timeout, headers)

        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMResponseError(
                f"Unexpected API response structure: {e}. "
                f"Response keys at top level: {list(result.keys()) if isinstance(result, dict) else type(result)}"
            )

        if content is None:
            raise LLMResponseError(
                "LLM returned a null response. The model may have "
                "filtered or refused the request."
            )

        return strip_llm_noise(content)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BACKEND_MAP = {
    "ollama": OllamaBackend,
    "openai": OpenAIBackend,
    "llamacpp": OpenAIBackend,
}

BACKEND_DEFAULTS = {
    "ollama": {
        "url": "http://localhost:11434",
        "model": "llama3.2",
    },
    "openai": {
        "url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "llamacpp": {
        "url": "http://localhost:8080",
        "model": "",
    },
}

BACKEND_LABELS = {
    "ollama": "Ollama",
    "openai": "OpenAI Compatible",
    "llamacpp": "llama.cpp",
}


def create_backend(provider, base_url, model, timeout=60, api_key=None):
    """Factory: return an ``LLMBackend`` instance for *provider*.

    Falls back to provider-specific defaults when *base_url* or
    *model* are empty.
    """
    if provider not in BACKEND_MAP:
        available = ", ".join(BACKEND_MAP.keys())
        raise ValueError(
            f"Unknown provider: {provider!r}. "
            f"Supported: {available}"
        )

    if not base_url:
        base_url = BACKEND_DEFAULTS.get(provider, {}).get("url", "")

    if not model:
        model = BACKEND_DEFAULTS.get(provider, {}).get("model", "")

    cls = BACKEND_MAP[provider]

    if cls is OllamaBackend:
        return OllamaBackend(base_url, model, timeout)

    return OpenAIBackend(base_url, model, timeout, api_key)
