import base64
from typing import Dict, Any

from mistralai import Mistral

from ..base import OCRSystem


class MistralOCR(OCRSystem):
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.model = Mistral(api_key=self.config.get('api_key'))

    def extract_raw_output(self, image_path: str) -> Dict[str, Any]:
        with open(image_path, 'rb') as img_file:
            img = base64.b64encode(img_file.read()).decode('utf-8')

        ocr_response = self.model.ocr.process(
            model=self.config.get('model', 'mistral-ocr-2505-completion'),
            document={
                'type': 'image_url',
                'image_url': f'data:image/jpeg;base64,{img}',
            },
        )
        return ocr_response.model_dump()
