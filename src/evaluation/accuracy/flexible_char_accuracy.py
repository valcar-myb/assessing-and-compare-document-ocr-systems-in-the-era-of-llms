"""
Python reimplementation of FlexCharacterAccuracy (PRImA Research Lab).
Flexible character accuracy with non-sequential line-by-line matching.
Edit distance is computed with rapidfuzz.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from rapidfuzz.distance import Levenshtein as _Levenshtein

# Edit distance costs: (insertion, deletion, substitution)
COST_INS1_DEL1_SUBST1 = (1, 1, 1)
COST_INS1_DEL1_SUBST2 = (1, 1, 2)

# Optional reduced grid: FLEX_FAST_GRID=1 for ~16 combinations (faster).
# Default: full grid (768, as in the original implementation).
_FLEX_FAST_GRID = os.environ.get("FLEX_FAST_GRID", "").lower() in ("1", "true", "yes")


def normalize_text(text: str) -> str:
    """Remove \\r\\n and \\n from the text (as in the Java version)."""
    if not text:
        return ""
    return text.replace("\r\n", "").replace("\n", "")


def split_into_lines(text: str) -> List[str]:
    """Split on \\r\\n or \\n; return only non-empty lines."""
    if not text:
        return []
    if "\r\n" in text:
        parts = text.split("\r\n")
    else:
        parts = text.split("\n")
    return [p for p in parts if p is not None and p != ""]


def edit_distance(
    s1: str,
    s2: str,
    cost_ins: int = 1,
    cost_del: int = 1,
    cost_sub: int = 1,
) -> int:
    """Levenshtein distance with configurable costs (via rapidfuzz)."""
    return _Levenshtein.distance(s1, s2, weights=(cost_ins, cost_del, cost_sub))


def character_accuracy(
    expected: str,
    result: str,
    cost_function: str = "INS1_DEL1_SUBST1",
) -> float:
    """
    Classic character accuracy on normalized text.
    accuracy = max(0, (len(expected) - edit_dist) / len(expected)).
    """
    exp = normalize_text(expected)
    res = normalize_text(result)
    costs = COST_INS1_DEL1_SUBST1 if cost_function == "INS1_DEL1_SUBST1" else COST_INS1_DEL1_SUBST2
    cost_ins, cost_del, cost_sub = costs
    if len(exp) == 0:
        return 0.0
    dist = edit_distance(exp, res, cost_ins, cost_del, cost_sub)
    acc = (len(exp) - dist) / len(exp)
    return max(0.0, acc)


@dataclass
class MatchResult:
    """Result of the best local match (sliding substring)."""
    min_edit_dist: int
    substring_pos: int
    substring_length: int
    length_diff: int

    def calc_penalty(
        self,
        edit_dist_coeff: int,
        length_diff_coeff: int,
        offset_coeff: int,
        length_coeff: int,
    ) -> int:
        """Penalty formula (as in the Java version)."""
        if self.length_diff <= 1:
            offset = 0
        else:
            offset = self.length_diff // 2 - abs(
                self.substring_pos - self.length_diff // 2
            )
        return (
            self.min_edit_dist * edit_dist_coeff
            + self.length_diff * length_diff_coeff
            + offset * offset_coeff
            - self.substring_length * length_coeff
        )


def _get_costs(cost_function: str) -> Tuple[int, int, int]:
    if cost_function == "INS1_DEL1_SUBST2":
        return COST_INS1_DEL1_SUBST2
    return COST_INS1_DEL1_SUBST1


def calculate_best_edit_distance(
    expected: str,
    result: str,
    cost_function: str,
    cache: Dict[str, Dict[str, MatchResult]],
) -> MatchResult:
    """
    For each window of the short string's length over the long string,
    compute the edit distance; return the MatchResult with the minimum
    distance. In Python we use correct slices (expected[i:i+len(result)])
    to avoid the Java bug (substring with length-1).
    """
    cache_exp = cache.get(expected)
    if cache_exp is not None:
        cached = cache_exp.get(result)
        if cached is not None:
            return cached
    else:
        cache[expected] = {}

    cost_ins, cost_del, cost_sub = _get_costs(cost_function)
    res = MatchResult(
        min_edit_dist=0,
        substring_pos=0,
        substring_length=0,
        length_diff=0,
    )

    if len(expected) > len(result):
        res.length_diff = len(expected) - len(result)
        res.substring_length = len(result)
        min_edit_dist = 10 ** 9
        min_pos = -1
        for i in range(0, res.length_diff + 1):
            sub = expected[i : i + len(result)]
            dist = edit_distance(result, sub, cost_ins, cost_del, cost_sub)
            if dist < min_edit_dist:
                min_edit_dist = dist
                min_pos = i
        res.min_edit_dist = min_edit_dist
        res.substring_pos = min_pos
    elif len(result) > len(expected):
        res.length_diff = len(result) - len(expected)
        res.substring_length = len(expected)
        min_edit_dist = 10 ** 9
        min_pos = -1
        for i in range(0, res.length_diff + 1):
            sub = result[i : i + len(expected)]
            dist = edit_distance(sub, expected, cost_ins, cost_del, cost_sub)
            if dist < min_edit_dist:
                min_edit_dist = dist
                min_pos = i
        res.min_edit_dist = min_edit_dist
        res.substring_pos = min_pos
    else:
        res.min_edit_dist = edit_distance(
            result, expected, cost_ins, cost_del, cost_sub
        )
        res.substring_length = len(result)
        res.length_diff = 0
        res.substring_pos = 0

    cache[expected][result] = res
    return res


def _sort_by_length(lines: List[str]) -> None:
    """Sort in place by decreasing length."""
    lines.sort(key=len, reverse=True)


def calculate_flex_edit_distance(
    expected_lines: List[str],
    result_lines: List[str],
    edit_dist_coeff: int,
    length_diff_coeff: int,
    offset_coeff: int,
    length_coeff: int,
    cost_function: str,
    cache: Dict[str, Dict[str, MatchResult]],
) -> Tuple[int, int, int]:
    """
    Compute the flex edit distance between the two lists of lines.
    Returns (expected_chars, result_chars, total_edit_dist).
    """
    total_edit_dist = 0
    expected_number_of_chars = sum(len(s) for s in expected_lines)
    result_number_of_chars = sum(len(s) for s in result_lines)

    exp_lines = list(expected_lines)
    res_lines = list(result_lines)
    _sort_by_length(exp_lines)

    deletion_penalty = 1  # as in the Java version for both cost functions
    insert_penalty = 1

    while exp_lines:
        min_edit_dist = 10 ** 9
        min_penalty = 10 ** 9
        min_result_idx = -1
        min_expected_idx = -1
        min_substring_pos = 0

        for i, expected_line in enumerate(exp_lines):
            for j, result_line in enumerate(res_lines):
                match_res = calculate_best_edit_distance(
                    expected_line,
                    result_line,
                    cost_function,
                    cache,
                )
                penalty = match_res.calc_penalty(
                    edit_dist_coeff,
                    length_diff_coeff,
                    offset_coeff,
                    length_coeff,
                )
                if penalty < min_penalty:
                    min_penalty = penalty
                    min_edit_dist = match_res.min_edit_dist
                    min_expected_idx = i
                    min_result_idx = j
                    min_substring_pos = match_res.substring_pos

        if min_result_idx < 0:
            break

        expected_line = exp_lines[min_expected_idx]
        result_line = res_lines[min_result_idx]

        if len(expected_line) > len(result_line):
            left = (
                expected_line[:min_substring_pos].strip()
                if min_substring_pos > 0
                else None
            )
            right_start = min_substring_pos + len(result_line)
            right = (
                expected_line[right_start:].strip()
                if right_start < len(expected_line)
                else None
            )
            if left:
                exp_lines.append(left)
            if right:
                exp_lines.append(right)
        elif len(result_line) > len(expected_line):
            left = (
                result_line[:min_substring_pos].strip()
                if min_substring_pos > 0
                else None
            )
            right_start = min_substring_pos + len(expected_line)
            right = (
                result_line[right_start:].strip()
                if right_start < len(result_line)
                else None
            )
            if left:
                res_lines.append(left)
            if right:
                res_lines.append(right)

        res_lines.pop(min_result_idx)
        total_edit_dist += (
            min_edit_dist if min_edit_dist != 10 ** 9 else len(expected_line)
        )
        exp_lines.pop(min_expected_idx)
        _sort_by_length(exp_lines)

    for line in res_lines:
        total_edit_dist += len(line) * deletion_penalty
    for line in exp_lines:
        total_edit_dist += len(line) * insert_penalty

    return (
        expected_number_of_chars,
        result_number_of_chars,
        total_edit_dist,
    )


def _coefficient_grid(fast: bool) -> List[Tuple[int, int, int, int]]:
    """Grid of (edit_dist_coeff, length_diff_coeff, offset_coeff, length_coeff)."""
    if fast:
        # Reduced grid: ~16 combinations, result very close to the full grid
        edit_range = [15, 25]  # was 15, 20, 25, 30
        length_diff_range = [0, 12]  # was 0, 3, 6, ..., 21
        offset_range = [0, 2]  # was 0, 1, 2, 3
        length_range = [0, 3]  # was 0, 1, ..., 5
    else:
        edit_range = list(range(15, 31, 5))
        length_diff_range = list(range(0, 24, 3))
        offset_range = list(range(0, 4))
        length_range = list(range(0, 6))
    return [
        (e, ld, o, l)
        for e in edit_range
        for ld in length_diff_range
        for o in offset_range
        for l in length_range
    ]


def evaluate(
    expected: str,
    result: str,
    cost_function: str = "INS1_DEL1_SUBST1",
    fast_grid: bool = None,
) -> Dict[str, float]:
    """
    Evaluate Flexible Character Accuracy between expected (ground truth) and
    result (OCR output). Returns a dict with:
      - flex_character_accuracy: float in [0, 1]
      - chars_in_ground_truth: int
      - chars_in_result: int
    fast_grid: if True, use the reduced grid (~16 combinations instead of 768).
    Default False (full grid, as in the original). Can also be set via the
    FLEX_FAST_GRID=1 environment variable.
    """
    if fast_grid is None:
        fast_grid = _FLEX_FAST_GRID
    match_cache: Dict[str, Dict[str, MatchResult]] = {}
    max_char_acc = 0.0
    expected_number_of_chars = 0
    result_number_of_chars = 0

    # Classic character accuracy (lower bound)
    max_char_acc = character_accuracy(expected, result, cost_function)

    grid = _coefficient_grid(fast_grid)
    for edit_dist_coeff, length_diff_coeff, offset_coeff, length_coeff in grid:
        exp_lines = split_into_lines(expected)
        res_lines = split_into_lines(result)
        if len(expected) > 0 or len(result) > 0:
            exp_chars, res_chars, edit_dist = (
                calculate_flex_edit_distance(
                    exp_lines,
                    res_lines,
                    edit_dist_coeff,
                    length_diff_coeff,
                    offset_coeff,
                    length_coeff,
                    cost_function,
                    match_cache,
                )
            )
            expected_number_of_chars = exp_chars
            result_number_of_chars = res_chars
            character_accuracy_val = 0.0
            if expected_number_of_chars > 0:
                character_accuracy_val = (
                    expected_number_of_chars - edit_dist
                ) / expected_number_of_chars
                character_accuracy_val = max(0.0, character_accuracy_val)
            if character_accuracy_val > max_char_acc:
                max_char_acc = character_accuracy_val

    return {
        "flex_character_accuracy": max_char_acc,
        "chars_in_ground_truth": expected_number_of_chars,
        "chars_in_result": result_number_of_chars,
    }


if __name__ == "__main__":
    # Quick test
    exp = "Hello world\nSecond line"
    res = "Hello world\nSecond line"
    out = evaluate(exp, res)
    print("Expected ~1.0:", out)

    exp2 = "abc\ndef"
    res2 = "abx\ndeef"
    out2 = evaluate(exp2, res2)
    print("With errors:", out2)
