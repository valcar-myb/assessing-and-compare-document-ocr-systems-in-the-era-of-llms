# Setup

This section documents the environment, dependencies, credentials, and dataset
preparation needed to reproduce the evaluation.

## Experimental environment

All experiments were conducted on a dedicated AWS cloud instance:

| Component | Specification |
|---|---|
| Instance | AWS `g5.xlarge` |
| GPU | 1× NVIDIA A10 (24 GB VRAM) |
| CPU / RAM | 4 vCPUs / 16 GB RAM |
| OS | Ubuntu 22.04 |
| CUDA | 12.2 |

API-based services were accessed from the AWS Frankfurt region instance;
open-source OCR engines and LLMs were run locally on the same instance.

## Dependencies

Shared dependencies are listed in [`requirements.txt`](../requirements.txt):

```bash
pip install -r requirements.txt
```

System-specific dependencies (OCR engines, cloud SDKs, LLM clients) are
installed per system. Recommended isolation:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Accuracy evaluation tool: ocreval

Character- and word-level accuracy (CA, WA) are computed with
[`ocreval`](https://github.com/eddieantonio/ocreval), a UTF-8–compatible
reimplementation of the ISRI OCR evaluation tools.

```bash
# macOS
brew install eddieantonio/eddieantonio/ocreval

# Linux: build from source (see the ocreval repository for prerequisites)
```

See [Metrics](metrics.md) for how the tool is used.

## Commercial services: credentials

Each commercial service is accessed through its official Python SDK. Configure
credentials via environment variables (recommended) or the provider's standard
mechanism.

| Provider | Service(s) | Credentials |
|---|---|---|
| AWS | Textract | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` |
| Microsoft Azure | AI Vision, Document Intelligence | endpoint + API key |
| Google Cloud | Vision, Document AI | service-account JSON (`GOOGLE_APPLICATION_CREDENTIALS`) |
| OpenAI | GPT-4o | `OPENAI_API_KEY` |
| Google | Gemini 2.0 Flash | API key |
| Anthropic | Claude Haiku 4.5 | `ANTHROPIC_API_KEY` |
| Mistral AI | Mistral Document AI | `MISTRAL_API_KEY` |

Deployment regions used for the commercial OCR services:

| Service | Region |
|---|---|
| AWS Textract | eu-central-1 (Frankfurt) |
| Azure (Vision, Document Intelligence) | West Europe (Netherlands) |
| Google Cloud (Vision, Document AI) | Global |

All commercial services were used with standard pay-as-you-go accounts at the
default service tier, one request per document image.

## Open-source LLMs: vLLM server

Open-source multimodal LLMs are served locally with the **vLLM** inference
engine (v0.4.2) in Docker containers based on `nvidia/cuda:12.2`, exposing an
OpenAI-compatible Chat Completions API. Models are pulled from the Hugging Face
Hub (see [Systems](systems.md) for the exact repositories). Decoding uses
provider defaults except `temperature = 0` for deterministic generation.

## Dataset preparation

Four public datasets are used. Each provides line-level ground truth, aggregated
into full-page text files for evaluation. Official train/test splits are used.

| Dataset | Document Type | Test size | Notes |
|---|---|---|---|
| SROIE | Printed receipts | 347 | Transcriptions normalized to lowercase |
| IAM | Handwritten text | 232 | Printed prompt region cropped out; only handwriting kept |
| FUNSD | Noisy scanned forms | 50 | Word-level transcriptions + boxes → full-page text |
| FOX | Dense text (EN + CN) | 212 | Evaluation-only; page-level text in reading order |

Outputs are normalized before metric computation (character whitelist for
English datasets; character-level evaluation for the Chinese subset of FOX);
details in [Metrics](metrics.md).
