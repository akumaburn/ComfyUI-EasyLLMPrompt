"""Prompt construction from structured scene description fields.

The default system prompt tells the LLM how to behave as an
SDXL prompt compiler.  The user message is a simple key-value
list of the fields the user filled in.
"""

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert Stable Diffusion XL prompt compiler.\n\n"
    "Convert the provided structured scene description into a "
    "single optimized SDXL positive prompt.\n\n"
    "Rules:\n"
    "* Output ONE comma-separated prompt only.\n"
    "* Do not explain anything.\n"
    "* Do not use sentences.\n"
    "* Do not use markdown.\n"
    "* Do not number items.\n"
    "* Never repeat concepts.\n"
    "* Never invent objects.\n"
    "* Never contradict user input.\n"
    "* Prioritize realism unless Notes specify another style.\n"
    "* SDXL prioritises the beginning of the prompt — the most critical concepts MUST appear first.\n"
    "* Order concepts from most important to least important.\n\n"
    "Use this strict order (most to least important):\n"
    "1. Subject\n"
    "2. Physical appearance\n"
    "3. Clothing\n"
    "4. Pose\n"
    "5. Action\n"
    "6. Expression\n"
    "7. Environment\n"
    "8. Time of day\n"
    "9. Lighting\n"
    "10. Composition\n"
    "11. Camera\n"
    "12. Quality descriptors\n\n"
    "IMPORTANT: This order is not a suggestion — it must be followed exactly because "
    "SDXL's CLIP model reads left-to-right and the first tokens carry the most weight "
    "in the generated image. The subject MUST be the very first thing in the prompt.\n\n"
    "Prefer concise descriptive tags rather than prose.\n"
    "Include appropriate photography terminology where applicable.\n"
    "Avoid unnecessary repetition."
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
