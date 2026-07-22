# Evaluated Systems

The study evaluates **16 systems**, organized into four categories:
*(i)* open-source OCR engines, *(ii)* commercial cloud-based OCR services,
*(iii)* commercial multimodal LLMs, and *(iv)* open-source multimodal LLMs.
Systems were selected for their widespread adoption, accessibility, and
suitability for a broad range of document processing tasks.

All systems were evaluated on **full-page text recognition** at the single-page
level, on the original inputs, without image pre-processing or output
post-processing. Heterogeneous outputs were converted to a plain-text
representation for evaluation (see [Metrics](metrics.md)).

## 1. Open-source OCR engines

Three widely adopted frameworks, each representing a distinct architectural
approach. They were accessed via their official Python libraries using default
configurations and pretrained weights.

| OCR Engine | Version | Detection Stage | Recognition Stage | GPU Support |
|---|---|---|---|---|
| Tesseract | 5.0 | Default (PSM=3) | LSTM-CTC (eng) | No |
| DocTR | 0.8.1 | DBNet (ResNet-50) | CRNN (VGG16-BN) | Yes |
| PaddleOCR | 2.3 | MobileNetV3 (PP-OCRv3) | SVTR-LCNet | Yes |

- **Tesseract** — accessed through `pytesseract`, CPU only.
- **DocTR** — official package with the PyTorch backend, GPU acceleration enabled.
- **PaddleOCR** — lightweight PP-OCRv3 configuration; evaluated in both CPU and GPU configurations.

DocTR and PaddleOCR were evaluated on both CPU and GPU; Tesseract on CPU only.

### PaddleOCR-FT (fine-tuned)

To assess domain adaptation, the SVTR-LCNet recognition module of PaddleOCR was
fine-tuned per dataset (separate models for SROIE, FUNSD, and IAM), initialized
from the official PP-OCRv3 pretrained weights, for 50 epochs. FOX is
evaluation-only (no line/word boxes for supervised fine-tuning).

## 2. Commercial OCR services

Fully managed cloud OCR services from Microsoft Azure, Google Cloud Platform,
and Amazon Web Services. Each was accessed through its official Python SDK, one
request per document image; synchronous APIs were used where available, while
Google Document AI and Azure Document Intelligence used asynchronous endpoints.

| Provider | Service | Target Use Case | Model/Version Control |
|---|---|---|---|
| Microsoft Azure | AI Vision (v4.0) | General-purpose images | Yes (version selectable) |
| Microsoft Azure | Document Intelligence | Documents | Yes (model selectable) |
| Google Cloud | Vision OCR | General-purpose images | Yes (model selectable) |
| Google Cloud | Document AI OCR | Documents | Yes (model selectable) |
| AWS | Textract | Documents | No (latest) |

Azure and Google allow selecting the model/version used for transcription,
whereas AWS Textract abstracts versioning details.

## 3. Commercial multimodal LLMs

Accessed through their official APIs and Python libraries. Where configurable,
`temperature = 0`; otherwise provider defaults were applied. Each image was
submitted as a single text–image completion request returning plain text.

| Model | Provider | Version | OCR Capabilities |
|---|---|---|---|
| GPT-4o | OpenAI | 2024-08-06 | Plain text |
| Gemini 2.0 Flash | Google | 2024-12 | Plain text + bounding boxes |
| Claude Haiku 4.5 | Anthropic | 2025-10-01 | Plain text |
| Mistral Document AI | Mistral AI | mistral-ocr-2505-completion | JSON/Markdown metadata |

## 4. Open-source multimodal LLMs

Executed locally with the **vLLM** inference engine (v0.4.2) in Docker
containers based on `nvidia/cuda:12.2`, served through the OpenAI-compatible
Chat Completions API. Default decoding parameters except `temperature = 0`.

| Model | Provider | Parameters | Vision Encoder | OCR Capabilities |
|---|---|---|---|---|
| Gemma 3 | Google | 4B | SigLIP ViT | Plain text |
| Qwen2.5-VL | Alibaba | 3B | ViT-L | Plain text + bounding boxes |
| PaddleOCR-VL | Baidu | 0.9B | NaViT (+ ERNIE-4.5-0.3B) | JSON/Markdown metadata |
| DeepSeek-OCR | DeepSeek | 3.3B | DeepEncoder (SAM/CLIP) | JSON/Markdown metadata |

Hugging Face repositories:
- Gemma 3 (4B-it): <https://huggingface.co/google/gemma-3-4b-it>
- Qwen2.5-VL (3B-Instruct): <https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct>
- PaddleOCR-VL (0.9B): <https://huggingface.co/PaddlePaddle/PaddleOCR-VL>
- DeepSeek-OCR: <https://huggingface.co/deepseek-ai/DeepSeek-OCR>

## Prompting

General-purpose multimodal LLMs (GPT-4o, Gemini 2.0 Flash, Claude Haiku 4.5,
Gemma 3, Qwen2.5-VL) were evaluated under two prompting conditions:

- **Simple instruction:** `Extract all visible text from the document image. Return plain text.`
- **Chain-of-Thought (CoT):** a multi-step prompt that guides the model to analyze
  layout, scan text regions, resolve ambiguous characters, and return only the
  final transcription.

Document-specialized models use task-specific prompts:

- **PaddleOCR-VL:** `OCR`
- **DeepSeek-OCR:** `<image>\nFree OCR.`
- **Mistral Document AI:** no configurable prompt interface.
