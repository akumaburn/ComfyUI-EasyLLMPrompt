"""ComfyUI-EasyLLMPrompt: structured scene description → SDXL prompt.

Node registration entry point read by ComfyUI on startup.
"""

from .easy_llm_prompt.node import EasyLLMPromptNode

NODE_CLASS_MAPPINGS = {
    "EasyLLMPromptNode": EasyLLMPromptNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyLLMPromptNode": "Easy LLM Prompt",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
