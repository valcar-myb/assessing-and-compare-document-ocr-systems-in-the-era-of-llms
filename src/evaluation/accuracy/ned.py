"""
Normalized Edit Distance (NED) e NED similarity per la valutazione OCR.

Definizioni (costi unitari per inserzione/cancellazione/sostituzione):

    NED     = edit_distance(pred, gt) / max(|pred|, |gt|)
    NED_sim = 1 - NED

NED_sim è in [0, 1] (1 = identico). Modulo standalone e plug-and-play:
opera direttamente su due stringhe (ground truth e output OCR).
Usa rapidfuzz se disponibile (più veloce), altrimenti fallback puro Python.
"""

from typing import Dict

try:
    from rapidfuzz.distance import Levenshtein as _RF_Levenshtein
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False


def _levenshtein_python(s1: str, s2: str) -> int:
    """Distanza di Levenshtein a costi unitari (puro Python, memoria O(min))."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)
    previous = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, start=1):
        current = [i]
        for j, c2 in enumerate(s2, start=1):
            cost = 0 if c1 == c2 else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def edit_distance(pred: str, gt: str) -> int:
    """Distanza di Levenshtein a costi unitari tra pred e gt."""
    if _RAPIDFUZZ_AVAILABLE:
        return _RF_Levenshtein.distance(pred, gt)
    return _levenshtein_python(pred, gt)


def normalized_edit_distance(pred: str, gt: str) -> float:
    """NED = edit_distance / max(|pred|, |gt|). Ritorna 0.0 se entrambe vuote."""
    denom = max(len(pred), len(gt))
    if denom == 0:
        return 0.0
    return edit_distance(pred, gt) / denom


def ned_similarity(pred: str, gt: str) -> float:
    """NED_sim = 1 - NED, in [0, 1]."""
    return 1.0 - normalized_edit_distance(pred, gt)


def evaluate(pred: str, gt: str) -> Dict[str, float]:
    """
    Valuta NED tra pred (output OCR) e gt (ground truth).
    Ritorna un dict con:
      - ned: float in [0, 1]
      - ned_sim: float in [0, 1] (1 = identico)
    """
    ned = normalized_edit_distance(pred, gt)
    return {"ned": ned, "ned_sim": 1.0 - ned}


if __name__ == "__main__":
    # Test rapido
    print("Expected ned_sim ~1.0:", evaluate("Hello world", "Hello world"))
    print("With errors:", evaluate("Helo wrld", "Hello world"))
