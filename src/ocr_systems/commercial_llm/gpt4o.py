"""
OpenAI GPT-4o vision OCR implementation.
"""

import base64
from typing import Dict, Any
from ..base import OCRSystem


class GPT4oOCR(OCRSystem):
    """OpenAI GPT-4o vision OCR implementation."""

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.model = None
        self._init_predictor()

    def _init_predictor(self):
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI as OpenAIClient
            self.model = OpenAIClient(api_key=self.config.get('api_key'))
        except ImportError:
            print("OpenAI package not installed. Please install with: pip install openai")
            raise
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            raise

    def extract_raw_output(self, image_path: str) -> Dict[str, Any]:
        """Extract raw output from GPT-4o vision."""
        try:
            with open(image_path, 'rb') as img_file:
                img = base64.b64encode(img_file.read()).decode('utf-8')

            response = self.model.chat.completions.create(
                model=self.config.get('model', 'gpt-4o-2024-08-06'),
                messages=[
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'text',
                                'text': self.config.get('prompt', 'Extract all visible text from this document image. Return only the text'),
                            },
                            {
                                'type': 'image_url',
                                'image_url': {'url': f'data:image/jpeg;base64,{img}'},
                            },
                        ],
                    }
                ],
                max_tokens=self.config.get('max_tokens', 4096),
                temperature=self.config.get('temperature', 0.0),
            )
            return response.model_dump()
        except Exception as e:
            print(f"Error extracting raw output with GPT-4o: {e}")
            return {}
