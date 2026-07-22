"""
Anthropic Claude Haiku vision OCR implementation.
"""

import base64
from typing import Dict, Any
from ..base import OCRSystem


class ClaudeHaikuOCR(OCRSystem):
    """Anthropic Claude Haiku vision OCR implementation."""

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.model = None
        self._init_predictor()

    def _init_predictor(self):
        """Initialize the Anthropic Claude client."""
        try:
            import anthropic
            self.model = anthropic.Anthropic(api_key=self.config.get('api_key'))
        except ImportError:
            print("Anthropic package not installed. Please install with: pip install anthropic")
            raise
        except Exception as e:
            print(f"Error initializing Anthropic Claude client: {e}")
            raise

    def extract_raw_output(self, image_path: str) -> Dict[str, Any]:
        """Extract raw output from Claude Haiku vision."""
        try:
            with open(image_path, 'rb') as img_file:
                img = base64.b64encode(img_file.read()).decode('utf-8')

            # Claude only accepts jpeg, png, gif, webp.
            img_type = image_path.split('.')[-1].lower()
            if img_type not in ['jpeg', 'png', 'gif', 'webp']:
                img_type = 'jpeg'

            prompt = self.config.get('prompt', 'Extract all visible text from this document image. Return only the text')

            response = self.model.messages.create(
                model=self.config.get('model', 'claude-haiku-4-5'),
                max_tokens=self.config.get('max_tokens', 4096),
                temperature=self.config.get('temperature', 0.0),
                messages=[
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'image',
                                'source': {
                                    'type': 'base64',
                                    'media_type': f'image/{img_type}',
                                    'data': img,
                                },
                            },
                            {'type': 'text', 'text': prompt},
                        ],
                    }
                ],
            )
            return response.model_dump()
        except Exception as e:
            print(f"Error extracting raw output with Claude Haiku: {e}")
            return {}
