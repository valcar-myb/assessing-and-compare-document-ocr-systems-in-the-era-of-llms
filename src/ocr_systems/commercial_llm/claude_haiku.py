import base64
from typing import Dict, Any

import anthropic

from ..base import OCRSystem


class ClaudeHaikuOCR(OCRSystem):
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.model = anthropic.Anthropic(api_key=self.config.get('api_key'))

    def extract_raw_output(self, image_path: str) -> Dict[str, Any]:
        with open(image_path, 'rb') as img_file:
            img = base64.b64encode(img_file.read()).decode('utf-8')

        # Claude only accepts jpeg, png, gif, webp.
        img_type = image_path.split('.')[-1].lower()
        if img_type not in ['jpeg', 'png', 'gif', 'webp']:
            img_type = 'jpeg'

        prompt = self.config.get(
            'prompt',
            'Extract all visible text from this document image. Return only the text',
        )

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
