from .utils import LLAMA_MODEL, clean_response, safe_ollama_chat


def correct_llama(noisy: str) -> str:
    """
    C2a: LLaMA-only correction for a single noisy clinical dialogue.
    """
    prompt = (
        "You will receive a noisy clinical conversation transcript.\n"
        "Fix only clear ASR transcription errors without changing meaning.\n"
        "Do NOT summarise. Output ONLY the corrected text.\n\n"
        f"{noisy}"
    )
    raw = safe_ollama_chat(LLAMA_MODEL, prompt)
    return clean_response(raw)

