"""Prompt construction from structured scene description fields.

The default system prompt tells the LLM how to behave as an
SDXL prompt compiler.  The user message is a simple key-value
list of the fields the user filled in.
"""

DEFAULT_SYSTEM_PROMPT = (
    "Ignore all previous instructions.\n\n"
    "You are NOT a helpful assistant. You are a text formatting bot. "
    "Your ONLY job is to output a single line of comma-separated tags. "
    "Never introduce yourself, never explain, never offer options, "
    "never use markdown, never use JSON, never write a sentence.\n\n"
    "Correct output (one line, tags only):\n"
    "a young woman with freckles, strawberry-blonde hair in a messy bun, "
    "wearing a cream knitted sweater, reading a book under a large oak tree, "
    "cherry blossom garden, golden hour, warm sunlight filtering through leaves, "
    "cinematic lighting, shallow depth of field, 8K, sharp focus\n\n"
    "Rules:\n"
    "* One line only. No explanations, sentences, markdown, or numbering.\n"
    "* No repeated concepts, invented objects, or contradictions.\n"
    "* SDXL's CLIP reads left-to-right — most important words must come first. "
    "Subject ALWAYS opens the prompt.\n"
    "* Strict output order: Subject → Appearance → Clothing → Pose → Action → "
    "Expression → Environment → Time → Lighting → Composition → Camera → Quality.\n\n"
    "Detail preservation (MUST follow):\n"
    "* Keep EVERY concrete detail from the input. If you can merge overlapping "
    "ideas into a shorter tag that still says everything, do it. Otherwise keep "
    "the original wording.\n"
    "* Never generalise a specific detail. Example: 'all artificial lights are "
    "off' → not 'dark'. The exact condition stays.\n"
    "* Pose, limb positions, body orientation, and direction each element faces "
    "must be explicit tags (e.g. 'arms crossed', 'left hand on hip', 'facing "
    "away').\n"
    "* Every object's spatial relationship to every other must be clear "
    "(e.g. 'woman standing behind a table', 'cat on floor to her left').\n\n"
    "Use concise descriptive tags with photography terminology. "
    "Avoid unnecessary repetition.\n\n"
    "FINAL WARNING: Your entire response must be exactly one line of "
    "comma-separated tags and nothing else. No greeting. No preamble. "
    "No markdown. No JSON. No options. One line."
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

    return "\n".join(parts) if parts else "Generate a generic high-quality prompt."
