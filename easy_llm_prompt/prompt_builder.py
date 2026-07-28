"""Prompt construction from structured scene description fields.

The default system prompt tells the LLM how to behave as an
SDXL prompt compiler.  The user message is a simple key-value
list of the fields the user filled in.
"""

DEFAULT_SYSTEM_PROMPT = (
    "Ignore all previous instructions.\n\n"
    "You are a text reformatter. You do not generate images. "
    "Do not ask for an image generation tool. "
    "You only convert structured descriptions into a single line of "
    "comma-separated tags for an image generation model.\n\n"
    "Rules:\n"
    "* Output must be exactly one line of comma-separated tags.\n"
    "* No greetings, explanations, markdown, JSON, numbering, or options.\n"
    "* Subject always comes first. Order: Subject → Appearance → Clothing → "
    "Pose → Action → Expression → Environment → Time → Lighting → Quality.\n"
    "* Every input detail must appear in the output — verbatim or implied. "
    "Never omit. Never generalise (e.g. 'all lights off' → not 'dark').\n"
    "* No invented details. No repeated concepts. No contradictions.\n\n"
    "Example:\n"
    "Input: Subject: a young woman with freckles | Place: cherry blossom garden | "
    "Time: golden hour | Action: reading under a tree\n"
    "Output: a young woman with freckles, reading a book under a large oak tree, "
    "cherry blossom garden, golden hour, warm sunlight filtering through leaves, "
    "cinematic lighting, shallow depth of field, 8K, sharp focus"
)


def build_system_prompt(custom_prompt=None):
    """Return the system prompt to use.

    If *custom_prompt* is provided and non-empty it replaces
    the default entirely.
    """
    if custom_prompt and custom_prompt.strip():
        return custom_prompt.strip()
    return DEFAULT_SYSTEM_PROMPT


def build_user_message(subject="", place="", time="", action="", notes=""):
    """Assemble the structured scene description.

    Only fields with non-whitespace content are included so the
    LLM does not see empty placeholders.
    """
    parts = []

    if subject.strip():
        parts.append(f"Subject: {subject.strip()}")
    if place.strip():
        parts.append(f"Place: {place.strip()}")
    if time.strip():
        parts.append(f"Time: {time.strip()}")
    if action.strip():
        parts.append(f"Action: {action.strip()}")
    if notes.strip():
        parts.append(f"Notes: {notes.strip()}")

    return "\n".join(parts) if parts else "Subject: generic high-quality scene"
