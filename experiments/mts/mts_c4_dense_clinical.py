import re
from typing import List

from .rag_corpus import dense_retrieve
from .utils import MISTRAL_MODEL, chunk_with_overlap, clean_response, safe_ollama_chat


def strip_clinical_notes(text: str) -> str:
    """
    Remove obvious clinical-note headings and commentary from an LLM response,
    keeping only dialogue-like lines.
    """
    if not text:
        return ""

    lines = text.splitlines()
    cleaned_lines: List[str] = []

    heading_prefixes = (
        "symptoms:",
        "diagnosis:",
        "history of patient:",
        "history:",
        "plan of action:",
        "assessment:",
        "impression:",
        "transcript",
        "---",
    )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        lower = stripped.lower()

        # Drop obvious headings and transcript markers
        if any(lower.startswith(p) for p in heading_prefixes):
            continue

        # Drop bullet-style list items
        if re.match(r"^[-*•]\s+", stripped):
            continue

        # Drop pure parenthetical commentary
        if lower.startswith("(") and lower.endswith(")"):
            continue

        # If the line looks like "Role: ..." keep only common speaker roles, drop others
        m = re.match(r"^([a-zA-Z ]+):", stripped)
        if m:
            role = m.group(1).strip().lower()
            if role not in {"doctor", "patient", "dr", "nurse"}:
                # Likely a note heading (e.g., "Symptoms:", "Plan:")
                continue

        cleaned_lines.append(stripped)

    return " ".join(cleaned_lines).strip()


def enforce_single_continuous_passage(text: str) -> str:
    """
    Keep only one continuous corrected transcript passage.
    If the model emits repeated blocks (e.g. multiple TRANSCRIPT sections),
    keep the segment with the strongest dialogue signal.
    """
    if not text:
        return ""

    parts = [p.strip() for p in re.split(r"\btranscript\s*:\s*", text, flags=re.IGNORECASE) if p.strip()]
    if not parts:
        return ""
    if len(parts) == 1:
        return re.sub(r"\s+", " ", parts[0]).strip()

    def score(segment: str) -> int:
        low = segment.lower()
        role_hits = low.count("doctor:") + low.count("patient:")
        return role_hits * 1000 + len(segment)

    best = max(parts, key=score)
    return re.sub(r"\s+", " ", best).strip()


def _truncate_to_reference_ratio(text: str, input_word_count: int, max_ratio: float = 1.5) -> str:
    """If output is much longer than input, keep only the first max_ratio * input words (avoids WER inflation from repetition)."""
    if not text or not input_word_count:
        return text
    words = text.split()
    cap = max(input_word_count, int(input_word_count * max_ratio))
    if len(words) <= cap:
        return text
    # Prefer cutting at a sentence boundary
    truncated = " ".join(words[:cap])
    last_period = truncated.rfind(". ")
    if last_period > cap * 0.5:  # keep sentence boundary if not too far back
        return truncated[: last_period + 1].strip()
    return truncated.strip()


def correct_dense_clinical(noisy: str,
                           passages: List[str],
                           index,
                           model) -> str:
    """
    C4-Den-Mat: Dense RAG over clinical notes (domain-matched) with Mistral.
    Ensures output is dialogue-only (doctor/patient turns). Uses strict length
    limits to prevent verbose/repeated output that inflates WER.
    """
    chunks = chunk_with_overlap(noisy)
    out_chunks: List[str] = []
    input_words = len(noisy.split())

    for ch in chunks:
        ch_words = len(ch.split())
        # Cap generation at ~1.3x chunk length to allow corrections but not repetition
        num_predict = min(2048, int(ch_words * 1.3) + 100)

        ctx = dense_retrieve(ch, passages, index, model)
        context = "\n---\n".join(ctx)
        prompt = (
            "You will receive a noisy clinical conversation excerpt and some clinical notes.\n"
            "Use the notes only to help with names and clinical terminology.\n"
            "Fix clear ASR transcription errors. Output ONLY the corrected dialogue for this excerpt.\n"
            "Do NOT add headings, labels, summaries, bullet points, or multiple transcript blocks.\n"
            "Do NOT repeat or rephrase beyond the excerpt. Keep similar length to the input.\n\n"
            f"CLINICAL NOTES:\n{context}\n\n"
            f"NOISY EXCERPT:\n{ch}\n\n"
            "Corrected dialogue only:"
        )
        raw = safe_ollama_chat(MISTRAL_MODEL, prompt, num_predict=num_predict)
        cleaned = clean_response(raw, preserve_newlines=False)
        cleaned = strip_clinical_notes(cleaned)
        cleaned = enforce_single_continuous_passage(cleaned)
        out_chunks.append(cleaned)

    result = " ".join(out_chunks).strip()
    return _truncate_to_reference_ratio(result, input_words, max_ratio=1.5)

