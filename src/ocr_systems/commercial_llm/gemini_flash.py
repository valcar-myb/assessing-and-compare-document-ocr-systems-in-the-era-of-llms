from typing import Dict, Any

from google import genai
from google.genai import types
from PIL import Image

from ..base import OCRSystem


class GeminiFlashOCR(OCRSystem):
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.model = genai.Client(api_key=self.config.get('api_key'))

    def extract_raw_output(self, image_path: str) -> Dict[str, Any]:
        img = Image.open(image_path)
        prompt = self.config.get(
            'prompt',
            'Extract all visible text from this document image. Return only the text',
        )

        response = self.model.models.generate_content(
            model=self.config.get('model', 'gemini-2.0-flash'),
            contents=[img, prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=self.config.get('max_tokens', 4096),
                temperature=self.config.get('temperature', 0.0),
            ),
        )
        return response.model_dump()
