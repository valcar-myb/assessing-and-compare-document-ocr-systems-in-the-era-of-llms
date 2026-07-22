"""
Wrapper around ocreval (ISRI OCR evaluation tools) for Character Accuracy (CA)
and Word Accuracy (WA), as used in the paper.

Requires ocreval installed and on the PATH (the `accuracy` and `wordacc` tools).
See docs/setup.md for installation.

Standalone, plug-and-play module: it operates on pairs of UTF-8 text files
(ground truth / OCR output). Both ocreval tools print a line
`<value>%  Accuracy`; the returned values are percentages (0-100), as reported
in the paper.

Command-line usage:
    python ocreval_wrapper.py <ground_truth.txt> <ocr_output.txt>
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

# First "<value>%  Accuracy" line of the report (character or word accuracy).
_ACCURACY_RE = re.compile(r"([-\d.]+)\s*%\s+Accuracy")


def _run_tool(tool: str, gt_path, pred_path, report_path: Optional[str] = None) -> str:
    """Run an ocreval tool on (correctfile=gt, generatedfile=pred) and return its report."""
    if shutil.which(tool) is None:
        raise RuntimeError(
            f"'{tool}' not found on PATH. Install ocreval (see docs/setup.md)."
        )
    args = [tool, str(gt_path), str(pred_path)]
    if report_path is not None:
        args.append(str(report_path))
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"'{tool}' failed (exit {proc.returncode}): {proc.stderr.strip()}")
    if report_path is not None:
        return Path(report_path).read_text(encoding="utf-8", errors="replace")
    return proc.stdout


def _parse_accuracy(report: str) -> float:
    """Extract the first accuracy percentage from an ocreval report."""
    match = _ACCURACY_RE.search(report)
    if match is None:
        raise RuntimeError("Could not extract accuracy from the ocreval report.")
    return float(match.group(1))


def character_accuracy(gt_path, pred_path, report_path: Optional[str] = None) -> float:
    """Character Accuracy (%) via ocreval's `accuracy` tool."""
    return _parse_accuracy(_run_tool("accuracy", gt_path, pred_path, report_path))


def word_accuracy(gt_path, pred_path, report_path: Optional[str] = None) -> float:
    """Word Accuracy (%) via ocreval's `wordacc` tool."""
    return _parse_accuracy(_run_tool("wordacc", gt_path, pred_path, report_path))


def evaluate(gt_path, pred_path) -> Dict[str, float]:
    """
    Evaluate CA and WA for a (ground truth, OCR output) pair.
    Returns a dict with:
      - char_accuracy: float (percentage 0-100)
      - word_accuracy: float (percentage 0-100)
    """
    return {
        "char_accuracy": character_accuracy(gt_path, pred_path),
        "word_accuracy": word_accuracy(gt_path, pred_path),
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ocreval_wrapper.py <ground_truth.txt> <ocr_output.txt>")
        sys.exit(1)
    result = evaluate(sys.argv[1], sys.argv[2])
    print(f"Character Accuracy: {result['char_accuracy']:.2f}%")
    print(f"Word Accuracy:      {result['word_accuracy']:.2f}%")
