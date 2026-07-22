"""
Wrapper per ocreval (ISRI OCR evaluation tools) per Character Accuracy (CA)
e Word Accuracy (WA), come usate nel paper.

Richiede ocreval installato e nel PATH (tool `accuracy` e `wordacc`).
Vedi docs/setup.md per l'installazione.

Modulo standalone e plug-and-play: opera su coppie di file di testo UTF-8
(ground truth / output OCR). Entrambi i tool ocreval stampano una riga
`<valore>%  Accuracy`; i valori restituiti sono percentuali (0-100), come
riportate nel paper.

Uso da riga di comando:
    python ocreval_wrapper.py <ground_truth.txt> <ocr_output.txt>
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

# Prima riga "<valore>%  Accuracy" del report (character o word accuracy).
_ACCURACY_RE = re.compile(r"([-\d.]+)\s*%\s+Accuracy")


def _run_tool(tool: str, gt_path, pred_path, report_path: Optional[str] = None) -> str:
    """Esegue un tool ocreval su (correctfile=gt, generatedfile=pred) e ne ritorna il report."""
    if shutil.which(tool) is None:
        raise RuntimeError(
            f"'{tool}' non trovato nel PATH. Installa ocreval (vedi docs/setup.md)."
        )
    args = [tool, str(gt_path), str(pred_path)]
    if report_path is not None:
        args.append(str(report_path))
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"'{tool}' fallito (exit {proc.returncode}): {proc.stderr.strip()}")
    if report_path is not None:
        return Path(report_path).read_text(encoding="utf-8", errors="replace")
    return proc.stdout


def _parse_accuracy(report: str) -> float:
    """Estrae la prima percentuale di accuracy dal report ocreval."""
    match = _ACCURACY_RE.search(report)
    if match is None:
        raise RuntimeError("Impossibile estrarre l'accuracy dal report ocreval.")
    return float(match.group(1))


def character_accuracy(gt_path, pred_path, report_path: Optional[str] = None) -> float:
    """Character Accuracy (%) via il tool `accuracy` di ocreval."""
    return _parse_accuracy(_run_tool("accuracy", gt_path, pred_path, report_path))


def word_accuracy(gt_path, pred_path, report_path: Optional[str] = None) -> float:
    """Word Accuracy (%) via il tool `wordacc` di ocreval."""
    return _parse_accuracy(_run_tool("wordacc", gt_path, pred_path, report_path))


def evaluate(gt_path, pred_path) -> Dict[str, float]:
    """
    Valuta CA e WA per una coppia (ground truth, output OCR).
    Ritorna un dict con:
      - char_accuracy: float (percentuale 0-100)
      - word_accuracy: float (percentuale 0-100)
    """
    return {
        "char_accuracy": character_accuracy(gt_path, pred_path),
        "word_accuracy": word_accuracy(gt_path, pred_path),
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python ocreval_wrapper.py <ground_truth.txt> <ocr_output.txt>")
        sys.exit(1)
    result = evaluate(sys.argv[1], sys.argv[2])
    print(f"Character Accuracy: {result['char_accuracy']:.2f}%")
    print(f"Word Accuracy:      {result['word_accuracy']:.2f}%")
