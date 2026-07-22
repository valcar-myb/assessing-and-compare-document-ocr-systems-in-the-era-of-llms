"""
Google Gemini Flash vision OCR implementation.
"""

from typing import Dict, Any
from ..base import OCRSystem


class GeminiFlashOCR(OCRSystem):
    """Google Gemini Flash vision OCR implementation."""

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.model = None
        self._init_predictor()

    def _init_predictor(self):
        """Initialize the Google Gemini client."""
        try:
            from google import genai
            self.model = genai.Client(api_key=self.config.get('api_key'))
            self.genai = genai
        except ImportError:
            print("Google GenAI package not installed. Please install with: pip install google-genai")
            raise
        except Exception as e:
            print(f"Error initializing Google Gemini client: {e}")
            raise

    def extract_raw_output(self, image_path: str) -> Dict[str, Any]:
        """Extract raw output from Gemini Flash vision."""
        try:
            from PIL import Image
            from google.genai import types

            img = Image.open(image_path)
            prompt = self.config.get('prompt', 'Extract all visible text from this document image. Return only the text')

            response = self.model.models.generate_content(
                model=self.config.get('model', 'gemini-2.0-flash'),
                contents=[img, prompt],
                config=types.GenerateContentConfig(
                    max_output_tokens=self.config.get('max_tokens', 8192),
                    temperature=self.config.get('temperature', 0.0),
                ),
            )
            return response.model_dump()
        except Exception as e:
            print(f"Error extracting raw output with Gemini Flash: {e}")
            return {}
