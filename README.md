# Assessing and Comparing Document OCR Systems in the Era of Large Language Models

Companion repository for the paper *"Assessing and Comparing Document OCR
Systems in the Era of Large Language Models"* (submitted to *Information
Processing and Management*, Elsevier).

This repository documents the experimental evaluation to ensure reproducibility
and to facilitate future benchmarking and model extensions. It benchmarks **16
OCR and multimodal systems** across four categories — open-source OCR engines,
commercial OCR services, commercial multimodal LLMs, and open-source multimodal
LLMs — on four public datasets (SROIE, IAM, FUNSD, FOX EN/CN), along three
dimensions: **transcription accuracy**, **processing time**, and **monetary cost**.

## Documentation

The repository is organized around four self-contained areas:

- [Systems](docs/systems.md) — the evaluated systems, with versions/models and access mode.
- [Metrics](docs/metrics.md) — accuracy metrics (CA, WA, NED_sim, FCA), time, and cost.
- [Costs](docs/costs.md) — pricing models and per-page/per-token cost breakdowns.
- [Setup](docs/setup.md) — environment, dependencies, credentials, and dataset preparation.

## Repository structure

```
README.md            # this file (hub)
requirements.txt     # shared Python dependencies
LICENSE
docs/                # descriptive documentation (systems, metrics, costs, setup)
```

## Citation

```bibtex
@article{ocr_llm_assessment,
  title   = {Assessing and Comparing Document OCR Systems in the Era of Large Language Models},
  journal = {Information Processing and Management},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE).
