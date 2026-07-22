# Monetary Costs

Monetary cost is one of the three evaluation dimensions (together with accuracy
and processing time). This section documents the pricing models and the
cost figures used in the paper. The goal is **not** to suggest cost-optimization
strategies, but to report the expenses effectively incurred during
experimentation.

All monetary values are derived from the official pricing pages of the
respective providers **as of October 2025** and are subject to change. Costs and processing time are
reported for the **simple-prompt** condition only; the CoT prompt is
operationally equivalent (comparable output length, only a slightly longer
input prompt).

Pricing models differ by technology:

- **Commercial OCR services** — *page-based* pricing.
- **Multimodal LLMs** — *token-based* pricing (input and output tokens).
- **Open-source solutions** — no direct charge; cost is estimated from the
  infrastructure required to run them.

Dataset sizes used for aggregate costs: FUNSD (50), IAM (232), SROIE (347),
FOX (212) — 841 pages in total.

## 1. Page-based pricing

Commercial OCR APIs are highly uniform, ranging between \$0.0010 and \$0.0015
per page. Mistral Document AI is included here as the only next-generation
system offering a flat per-page scheme.

| System | Price per page (USD) |
|---|---|
| Azure Vision | 0.0010 |
| Azure Document Intelligence | 0.0015 |
| Google Vision | 0.0015 |
| Google Document AI | 0.0015 |
| AWS Textract | 0.0015 |
| Mistral Document AI | 0.0020 |

For the 841 evaluated pages, the aggregate cost for page-based systems amounts
to about **\$0.84**, with Azure Vision being the most affordable option.

## 2. Token-based pricing

Multimodal LLMs (Gemini 2.0 Flash, Claude Haiku 4.5, GPT-4o) are billed per
token, with separate input/output rates. The per-page cost accounts for all
input token components (image tokens, system prompt, user instruction) and
output tokens, following each provider's pricing structure.

**Aggregate cost by dataset (USD):**

| Provider | FUNSD (50) | IAM (232) | SROIE (347) | FOX (212) | Total |
|---|---|---|---|---|---|
| Gemini 2.0 Flash | 0.0235 | 0.0601 | 0.1250 | 0.1160 | 0.3246 |
| Claude Haiku 4.5 | 0.1548 | 0.4701 | 1.0597 | 1.3876 | 3.0722 |
| GPT-4o | 0.4330 | 1.3622 | 3.6088 | 3.9925 | 9.3965 |

Gemini 2.0 Flash is the most cost-efficient (\$0.32 total), while GPT-4o is the
most expensive (\$9.39); Claude Haiku 4.5 sits in between (\$3.07).

**Mean input / output tokens per page:**

| Model | FUNSD (50) | IAM (232) | SROIE (347) | FOX (212) |
|---|---|---|---|---|
| Gemini 2.0 Flash | 1,303 / 850 | 2,203 / 97 | 2,152 / 363 | 1,580 / 973 |
| Claude Haiku 4.5 | 1,041 / 411 | 1,543 / 97 | 1,316 / 348 | 1,231 / 1,063 |
| GPT-4o | 785 / 316 | 930 / 81 | 1,188 / 297 | 905 / 954 |

## 3. Open-source deployment (infrastructure)

Open-source systems entail infrastructure cost only. All experiments ran on an
AWS `g5.xlarge` instance (1× GPU A10, 4 vCPUs, 16 GB RAM):

| Item | Cost |
|---|---|
| On-demand | ≈ \$1.006 / hour (≈ \$8,800 / year continuous) |
| 1-year reserved plan | ≈ \$5,600 / year |

Per-page cost is computed as the hourly instance cost divided by the observed
mean throughput. Under the assumptions of this study, self-hosted open-source
deployments become cost-effective compared to commercial OCR APIs above roughly
**6 million pages/year**, and competitive with efficient LLM-based solutions
such as Gemini 2.0 Flash above roughly **20 million pages/year**. These
break-even points are illustrative estimates and depend on infrastructure
costs, throughput, deployment architecture, and provider pricing policies.

## 4. Projected costs at scale (1,000 pages)

Projected cost for processing 1,000 document pages under each pricing model,
based on the per-page and per-token rates observed in the experiments.

| System | Pricing Model | Cost per Page (USD) | Cost per 1k Pages (USD) |
|---|---|---|---|
| Mistral Document AI | Page-based | 0.0020 | 2.00 |
| Azure Doc. Intel. / Google Vision / Google Doc. AI / AWS Textract | Page-based | 0.0015 | 1.50 |
| Azure Vision | Page-based | 0.0010 | 1.00 |
| Gemini 2.0 Flash | Token-based | ≈ 0.0003 | ≈ 0.38 |
| Claude Haiku 4.5 | Token-based | ≈ 0.0036 | ≈ 3.65 |
| GPT-4o | Token-based | ≈ 0.011 | ≈ 11.17 |
| Open-source (g5.xlarge) | Infrastructure | ≈ 0.0002 | ≈ 0.17 |

For token-based systems, per-page costs are derived from the mean input/output
tokens observed across datasets, multiplied by the official per-token rates.
These figures are estimates: actual costs vary with document density, image
resolution, and model verbosity.

## Notes on economies of scale

Several providers offer volume-based discounts that can substantially reduce
per-page costs at high volume. For example, Google Document AI applies a reduced
rate of \$0.60 per 1,000 pages for monthly volumes above 5 million pages
(vs. the standard \$1.50 per 1,000 pages). Asynchronous batch APIs (e.g.,
Mistral, Gemini) can reduce token-based costs by up to 50% for workloads that do
not require real-time processing.

> All pricing figures were retrieved from official provider documentation in
> October 2025 and are subject to change.
