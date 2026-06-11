import json
import ollama


# Use a model large enough to reliably follow JSON schemas.
# qwen2.5:3b hallucinates too much for structured tool use.
# Recommended: qwen2.5:7b, mistral:7b, or llama3.2:latest
MODEL = "qwen3:8b"


def ask_llm(messages: list[dict], system_prompt: str) -> str:
    """
    Send a chat request to Ollama.
    Returns the raw response string (expected to be JSON by the agent).
    """
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    response = ollama.chat(
        model=MODEL,
        messages=full_messages,
        format="json",          # forces Ollama to return valid JSON every time
        options={
            "temperature": 0,   # deterministic — critical for tool-use reliability
            "num_ctx": 8192,
        },
    )

    return response["message"]["content"]