"""LRU cache for LLM request deduplication.

Identical requests (same system prompt, user message, temperature,
max tokens, seed, model, and backend) return cached results.
This avoids redundant API calls when the same node inputs are
evaluated multiple times.
"""

from collections import OrderedDict
import hashlib
import json
import threading


class PromptCache:
    """Thread-safe LRU cache keyed on full request parameters."""

    def __init__(self, max_size=100):
        self._cache = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()

    def _make_key(self, system_prompt, user_message, temperature,
                  max_tokens, seed, model, backend):
        """Produce a deterministic hash from the request parameters."""
        content = json.dumps(
            {
                "sp": system_prompt,
                "um": user_message,
                "t": temperature,
                "mt": max_tokens,
                "s": seed if seed is not None else -1,
                "m": model,
                "b": backend,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, system_prompt, user_message, temperature,
            max_tokens, seed, model, backend):
        """Return cached result or None."""
        key = self._make_key(
            system_prompt, user_message, temperature,
            max_tokens, seed, model, backend,
        )
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def set(self, system_prompt, user_message, temperature,
            max_tokens, seed, model, backend, result):
        """Store a result in the cache."""
        key = self._make_key(
            system_prompt, user_message, temperature,
            max_tokens, seed, model, backend,
        )
        with self._lock:
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def clear(self):
        """Remove all cached entries."""
        with self._lock:
            self._cache.clear()

    @property
    def size(self):
        """Number of entries currently cached."""
        with self._lock:
            return len(self._cache)
