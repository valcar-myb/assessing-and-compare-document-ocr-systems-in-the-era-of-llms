"""
Generic open-source VLM client via vLLM's OpenAI-compatible API.

Serves every open-source vision-language model evaluated in the paper:
Qwen2.5-VL, Gemma 3, PaddleOCR-VL and DeepSeek-OCR. The specific model is
selected with `hf_model_name` and the prompt with `prompt` (see
`config/systems.yaml`); PaddleOCR-VL and DeepSeek-OCR use their own
task prompts (`OCR` and `<image>\\nFree OCR.` respectively).
"""

import base64
import io
import requests
from typing import Dict, Any
from PIL import Image
from ..base import OCRSystem


class VLLMOpenAIOCR(OCRSystem):
    """Generic vLLM OCR client via the OpenAI-compatible chat completions API."""

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.api_url = config.get('api_url', 'http://localhost:8000/v1/chat/completions')
        self.hf_model_name = config.get('hf_model_name')

        if not self.hf_model_name:
            raise ValueError("hf_model_name is required for vLLM OCR systems")

    def extract_raw_output(self, image_path: str) -> Dict[str, Any]:
        """Extract raw output from a vLLM-served VLM."""
        try:
            with Image.open(image_path).convert('RGB') as img:
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            prompt = self.config.get('prompt', 'Extract all visible text from this document image. Return only the text')

            payload = {
                'model': self.hf_model_name,
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': prompt},
                            {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{image_b64}'}},
                        ],
                    }
                ],
                'max_tokens': self.config.get('max_tokens', 4096),
                'temperature': self.config.get('temperature', 0.0),
            }

            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error extracting raw output with vLLM ({self.hf_model_name}): {e}")
            return {}
