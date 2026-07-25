"""ComfyUI node: Easy LLM Prompt.

Converts a structured scene description (Subject, Place, Time,
Action, Notes) into an optimised SDXL positive prompt via an LLM.
"""

import os
import subprocess
import tempfile
import time as _time
import logging

from .cache import PromptCache
from .config import load_config
from .llm_backends import BACKEND_DEFAULTS, BACKEND_LABELS, create_backend
from .prompt_builder import build_system_prompt, build_user_message
from .utils import LLMResponseError

logger = logging.getLogger(__name__)

_cache = PromptCache(max_size=100)


def _run_shell(cmd):
    """Execute *cmd* as a bash script via a temp file (blocking).

    Using a temp file avoids the ``pkill -f`` problem: when a command is
    passed inline to ``/bin/sh -c <cmd>``, the pattern being grepped
    (e.g. ``llama-server``) appears in the shell's argv, so ``pkill -f``
    kills the shell itself.  Writing the script to a file keeps argv clean.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", prefix="easy_llm_", delete=False,
    ) as f:
        f.write("#!/usr/bin/env bash\n")
        f.write(cmd)
        f.write("\n")
        script_path = f.name
    try:
        os.chmod(script_path, 0o700)
        subprocess.run(["bash", script_path])
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------

BACKEND_CHOICES = list(BACKEND_DEFAULTS.keys())
BACKEND_DISPLAY = [BACKEND_LABELS[k] for k in BACKEND_CHOICES]


class EasyLLMPromptNode:
    """Convert a structured scene description into a single SDXL prompt.

    Input fields
    ------------
    **Subject** / **Place** / **Time** / **Action** / **Notes**
        Multiline text fields.  Fill in as many or as few as you need.

    **System Prompt** *(optional)*
        Override the built-in SDXL prompt-compiler instructions.
        Leave empty to use the default.

    **Backend** / **Base URL** / **Model**
        Which LLM server to talk to.  Defaults are persisted across
        ComfyUI sessions.

    **Temperature**
        Randomness control (0.0 = deterministic, 1.0 = balanced).
    """

    @classmethod
    def INPUT_TYPES(cls):
        cfg = load_config()

        return {
            "required": {
                "subject": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe the subject (e.g. 'a young woman with freckles')",
                }),
                "place": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe the environment (e.g. 'cherry blossom garden')",
                }),
                "time": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Time of day or era (e.g. 'golden hour', 'medieval')",
                }),
                "action": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "What is happening (e.g. 'reading a book under a tree')",
                }),
                "notes": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Additional notes or style guidance (e.g. 'Studio Ghibli style')",
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Override the built-in SDXL prompt compiler instructions",
                }),
                "backend": (BACKEND_CHOICES, {
                    "default": cfg.get("default_backend", "ollama"),
                }),
                "base_url": ("STRING", {
                    "default": cfg.get("default_base_url", "http://localhost:11434"),
                    "placeholder": "http://localhost:11434",
                }),
                "model": ("STRING", {
                    "default": cfg.get("default_model", "llama3.2"),
                    "placeholder": "llama3.2",
                }),
                "temperature": ("FLOAT", {
                    "default": cfg.get("default_temperature", 0.7),
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                }),
                "max_tokens": ("INT", {
                    "default": cfg.get("default_max_tokens", 512),
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "step": 1,
                    "tooltip": "-1 = random, any other value = deterministic",
                }),
                "before_run_shell": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Shell command to run BEFORE this node executes (blocking)",
                }),
                "after_run_shell": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Shell command to run AFTER this node executes (blocking)",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "debug_info")
    FUNCTION = "generate_prompt"
    CATEGORY = "prompt"
    OUTPUT_NODE = False

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate_prompt(self, subject="", place="", time="", action="",
                        notes="", system_prompt="", backend="ollama",
                        base_url="", model="", temperature=0.7,
                        max_tokens=512, seed=-1,
                        before_run_shell="", after_run_shell=""):
        """Execute the node: build prompts, call the LLM, return result."""
        start = _time.time()

        if before_run_shell.strip():
            _run_shell(before_run_shell)

        try:
            sys_prompt = build_system_prompt(system_prompt)
            user_msg = build_user_message(subject, place, time, action, notes)
            cache_seed = seed if seed >= 0 else None

            # ----- cache lookup -------------------------------------------------
            cached = _cache.get(
                sys_prompt, user_msg, temperature, max_tokens,
                cache_seed, model, backend,
            )
            if cached is not None:
                elapsed = _time.time() - start
                debug = (
                    f"Backend: {backend} | Model: {model} | "
                    f"Temp: {temperature} | Max tokens: {max_tokens} | "
                    f"Seed: {seed} | Time: {elapsed:.2f}s (cached)"
                )
                return (cached, debug)

            # ----- fall back to defaults for empty fields -----------------------
            if not base_url:
                base_url = BACKEND_DEFAULTS.get(backend, {}).get("url", "")
            if not model:
                model = BACKEND_DEFAULTS.get(backend, {}).get("model", "")

            # ----- LLM call -----------------------------------------------------
            try:
                cfg = load_config()
                llm = create_backend(backend, base_url, model,
                                     timeout=cfg.get("timeout", 60))

                result = llm.chat(
                    system_prompt=sys_prompt,
                    user_message=user_msg,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=cache_seed,
                )

                # cache the result
                _cache.set(
                    sys_prompt, user_msg, temperature, max_tokens,
                    cache_seed, model, backend, result,
                )

                elapsed = _time.time() - start
                debug = (
                    f"Backend: {backend} | Model: {model} | "
                    f"Temp: {temperature} | Max tokens: {max_tokens} | "
                    f"Seed: {seed} | Time: {elapsed:.2f}s"
                )
                return (result, debug)

            except (ValueError, LLMResponseError) as e:
                err = f"ERROR: {e}"
                return (err, f"Error: {err}")

            except Exception:
                logger.exception("Unexpected error in EasyLLMPromptNode")
                err = "ERROR: Unexpected error during prompt generation. Check ComfyUI log for details."
                return (err, f"Error: {err}")
        finally:
            if after_run_shell.strip():
                _run_shell(after_run_shell)

    # ------------------------------------------------------------------
    # Caching hint for ComfyUI
    # ------------------------------------------------------------------

    @classmethod
    def IS_CHANGED(cls, subject="", place="", time="", action="",
                   notes="", system_prompt="", backend="ollama",
                   base_url="", model="", temperature=0.7,
                   max_tokens=512, seed=-1,
                   before_run_shell="", after_run_shell="", **kwargs):
        """Control ComfyUI's workflow caching.

        * Random seed (``-1``): always re-execute (returns NaN).
        * Fixed seed: cache based on all input values.
        """
        if seed < 0:
            return float("NaN")

        return hash((
            subject, place, time, action, notes,
            system_prompt, backend, base_url, model,
            temperature, max_tokens, seed,
            before_run_shell, after_run_shell,
        ))
