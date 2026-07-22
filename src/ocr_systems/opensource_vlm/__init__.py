"""
Open-source vision-language model clients.

All four open-source VLMs evaluated in the paper (Qwen2.5-VL, Gemma 3,
PaddleOCR-VL and DeepSeek-OCR) are served locally through vLLM's
OpenAI-compatible API and share the single client in `vllm_client.py`. Each
model is selected via `hf_model_name` and its prompt through `prompt` in
`config/systems.yaml`.
"""
