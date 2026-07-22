# Assessing and Comparing Document OCR Systems in the Era of Large Language Models

Companion repository for the paper *"Assessing and Comparing Document OCR
Systems in the Era of Large Language Models"*.

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
LICENSE
config/              # declarative system configuration (systems.yaml)
docs/                # descriptive documentation (systems, metrics, costs, setup)
src/                 # evaluation metrics and OCR/LLM system clients
```

## License

Released under the [MIT License](LICENSE).
