"""Persistent configuration management.

Stores user settings in a JSON file within the ComfyUI user directory.
Settings persist across sessions so users don't have to re-enter them.
"""

import json
import os
import logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "default_backend": "ollama",
    "default_base_url": "http://localhost:11434",
    "default_model": "llama3.2",
    "default_temperature": 0.7,
    "default_max_tokens": 512,
    "timeout": 60,
    "cache_size": 100,
}

CONFIG_DIRNAME = "easy_llm_prompt"
CONFIG_FILENAME = "config.json"


def _get_config_dir():
    """Determine the configuration directory.

    Uses ComfyUI's user directory when available, otherwise
    falls back to ~/.config/easy_llm_prompt.
    """
    try:
        import folder_paths
        base = folder_paths.get_user_directory()
    except ImportError:
        base = os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, CONFIG_DIRNAME)


def _get_config_path():
    """Get the full path to the configuration file."""
    return os.path.join(_get_config_dir(), CONFIG_FILENAME)


def load_config():
    """Load configuration from disk, merging with defaults.

    Returns a dict with all keys from DEFAULT_CONFIG populated.
    Missing or corrupted files return defaults silently.
    """
    config = dict(DEFAULT_CONFIG)
    config_path = _get_config_path()

    if not os.path.exists(config_path):
        return config

    try:
        with open(config_path, "r") as f:
            user_config = json.load(f)
        config.update(user_config)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load config from %s: %s", config_path, e)

    return config


def save_config(config):
    """Persist configuration to disk.

    Creates the config directory if it does not exist.
    """
    config_dir = _get_config_dir()
    os.makedirs(config_dir, exist_ok=True)

    config_path = _get_config_path()
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.debug("Config saved to %s", config_path)
    except OSError as e:
        logger.error("Failed to save config to %s: %s", config_path, e)
