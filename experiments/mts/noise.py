import random
from typing import List


WORD_DROP_P = 0.08
WORD_SWAP_P = 0.05
CHAR_SUB_P = 0.03


def simulate_asr_noise(text: str,
                       drop_p: float = WORD_DROP_P,
                       swap_p: float = WORD_SWAP_P,
                       char_sub_p: float = CHAR_SUB_P,
                       rng: random.Random = random) -> str:
    """Simple word-drop / swap / char-sub noise model."""
    words: List[str] = text.split()
    if len(words) < 3:
        return text

    kept = [w for w in words if rng.random() > drop_p] or words
    i = 0
    while i < len(kept) - 1:
        if rng.random() < swap_p:
            kept[i], kept[i + 1] = kept[i + 1], kept[i]
            i += 2
        else:
            i += 1

    def sub_chars(w: str) -> str:
        if rng.random() >= char_sub_p or len(w) < 3:
            return w
        chars = list(w)
        idx = rng.randint(0, len(chars) - 1)
        replacements = {"f": "ph", "s": "z", "c": "k", "k": "c", "t": "d"}
        c = chars[idx].lower()
        if c in replacements:
            chars[idx] = replacements[c]
        return "".join(chars)

    kept = [sub_chars(w) for w in kept]
    return " ".join(kept)

