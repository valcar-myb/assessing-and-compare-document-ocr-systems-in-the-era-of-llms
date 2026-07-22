# Evaluation Metrics

Performance is assessed along three dimensions: **transcription accuracy**,
**processing time**, and **monetary cost** (the latter is documented in
[Costs](costs.md)). All outputs are converted to plain text and normalized
before accuracy computation.

## Transcription accuracy

Accuracy is assessed at the character and word levels. All metrics are reported
as similarity scores in `[0, 1]` (higher is better). For each system and
dataset, the paper reports 95% confidence intervals for the mean across pages.

### Character / Word Accuracy (CA, WA)

Character Error Rate (CER) and Word Error Rate (WER) are derived from the edit
(Levenshtein) distance — the minimum number of insertions, deletions, and
substitutions to transform the prediction into the ground truth:

$$CER = \frac{N_{ins} + N_{del} + N_{sub}}{N_{gt\_chars}} \qquad WER = \frac{N_{ins} + N_{del} + N_{sub}}{N_{gt\_words}}$$

Character and Word Accuracy follow directly:

$$CA = 1 - CER \qquad WA = 1 - WER$$

CA and WA are computed with the **ISRI Analytic Tools for OCR Evaluation**, via
[`ocreval`](https://github.com/eddieantonio/ocreval), a UTF-8–compatible
reimplementation of the ISRI toolkit. See [Setup](setup.md) for installation.

**Implementation:** [`src/evaluation/accuracy/ocreval_wrapper.py`](../src/evaluation/accuracy/ocreval_wrapper.py)
— a thin wrapper around the `accuracy` (CA) and `wordacc` (WA) tools; values are
returned as percentages (0–100).

```bash
python src/evaluation/accuracy/ocreval_wrapper.py ground_truth.txt ocr_output.txt
```

> **Limitation.** CA and WA operate on full-page linearized text and are
> therefore sensitive to the **serialization (reading) order** in which text
> regions are concatenated. A system that recognizes all text correctly but
> outputs it in a different order is unfairly penalized. This is especially
> relevant for form-like layouts (FUNSD) and multi-column documents (FOX).

### Normalized Edit Distance (NED_sim)

NED complements CA by normalizing the edit distance over the longer of the two
sequences, mitigating over-penalization when the prediction is much longer or
shorter than the ground truth:

$$NED(s_{pred}, s_{gt}) = \frac{N_{ins} + N_{del} + N_{sub}}{\max(|s_{pred}|, |s_{gt}|)}$$

Reported as a similarity score:

$$NED_{sim} = 1 - NED(s_{pred}, s_{gt})$$

**Implementation:** [`src/evaluation/accuracy/ned.py`](../src/evaluation/accuracy/ned.py)
(returns `ned_sim` in `[0, 1]`; edit distance computed with `rapidfuzz`).

### Flexible Character Accuracy (FCA)

To decouple character recognition from layout/serialization errors, we also
report **Flexible Character Accuracy** (Clausner et al., 2020), an
edit-distance-based measure designed to be **invariant to reading order**. FCA
compares substrings of the recognized and ground-truth text in a flexible way,
making it well suited to complex-layout documents where reading order cannot be
assumed consistent across systems.

FCA operates on **line units**:

- For OCR engines / OCR-oriented systems that natively return line-level
  transcriptions, the provided lines are used.
- For multimodal LLMs that return plain text, line units are derived from
  newline characters (`\n`) in the generated output.

**Implementation:** [`src/evaluation/accuracy/flexible_char_accuracy.py`](../src/evaluation/accuracy/flexible_char_accuracy.py)
— a Python reimplementation of PRImA's Flexible Character Accuracy. It performs
non-sequential line-to-line matching over a grid of penalty coefficients and
returns the best character accuracy. Edit distance is computed with `rapidfuzz`.

```python
from src.evaluation.accuracy.flexible_char_accuracy import evaluate

out = evaluate(expected=ground_truth_text, result=ocr_output_text)
print(out["flex_character_accuracy"])  # float in [0, 1]
```

Set the environment variable `FLEX_FAST_GRID=1` (or pass `fast_grid=True`) to use
a reduced coefficient grid for faster, approximate results.

Together, **CA**, **WA**, **NED_sim**, and **FCA** provide a complementary view
of transcription accuracy, capturing both character-level fidelity and
robustness to layout-induced serialization variability. The **FCA–CA gap** is
particularly informative: a large gap indicates that errors stem from reading
order inconsistencies rather than character misrecognition.

## Text normalization

To ensure fair comparison, all outputs are normalized before metric computation:

- **English datasets** (SROIE, FUNSD, IAM, FOX English subset): a conservative
  character whitelist is applied, retaining uppercase and lowercase Latin
  letters (A–Z, a–z), digits (0–9), whitespace, and standard punctuation
  ``. , ; : ! ? ( ) [ ] { } / \ @ # - + _ ' "``.
- **Chinese subset of FOX**: recognition is evaluated directly at the character
  level on the original Chinese Unicode text. Character-level metrics (CA,
  NED_sim, FCA) are reported; word-level metrics are omitted because Chinese has
  no explicit whitespace-delimited word boundaries.

## Processing time

For each dataset and system, per-page processing time (seconds) is measured, and
summary statistics (mean, median, min, max) with 95% confidence intervals are
reported. For cloud-based systems (commercial OCR and multimodal LLMs),
processing time includes both server-side inference and network round-trip
latency; for locally executed open-source systems it corresponds to model
inference time only. Reported latencies should therefore be interpreted as
end-to-end processing times rather than isolated inference times.

## Monetary cost

Cost is reported per pricing model (page-based, token-based, and open-source
infrastructure). See [Costs](costs.md) for the full breakdown.
