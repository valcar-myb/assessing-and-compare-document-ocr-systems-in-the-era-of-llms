"""
Normalized Edit Distance (NED) and NED similarity for OCR evaluation.

Definitions (unit costs for insertion/deletion/substitution):

    NED     = edit_distance(pred, gt) / max(|pred|, |gt|)
    NED_sim = 1 - NED

NED_sim lies in [0, 1] (1 = identical). Standalone, plug-and-play module:
it operates directly on two strings (ground truth and OCR output).
Edit distance is computed with rapidfuzz.
"""

from typing import Dict

from rapidfuzz.distance import Levenshtein as _Levenshtein


def edit_distance(pred: str, gt: str) -> int:
    """Unit-cost Levenshtein distance between pred and gt (via rapidfuzz)."""
    return _Levenshtein.distance(pred, gt)


def normalized_edit_distance(pred: str, gt: str) -> float:
    """NED = edit_distance / max(|pred|, |gt|). Returns 0.0 if both are empty."""
    denom = max(len(pred), len(gt))
    if denom == 0:
        return 0.0
    return edit_distance(pred, gt) / denom


def ned_similarity(pred: str, gt: str) -> float:
    """NED_sim = 1 - NED, in [0, 1]."""
    return 1.0 - normalized_edit_distance(pred, gt)


def evaluate(pred: str, gt: str) -> Dict[str, float]:
    """
    Evaluate NED between pred (OCR output) and gt (ground truth).
    Returns a dict with:
      - ned: float in [0, 1]
      - ned_sim: float in [0, 1] (1 = identical)
    """
    ned = normalized_edit_distance(pred, gt)
    return {"ned": ned, "ned_sim": 1.0 - ned}


if __name__ == "__main__":
    # Quick test
    print("Expected ned_sim ~1.0:", evaluate("Hello world", "Hello world"))
    print("With errors:", evaluate("Helo wrld", "Hello world"))
