from typing import Dict, Any

import boto3

from ..base import OCRSystem


class AWSTextractOCR(OCRSystem):
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.model = boto3.client(
            service_name='textract',
            aws_access_key_id=self.config.get('aws_access_key_id'),
            aws_secret_access_key=self.config.get('aws_secret_access_key'),
            region_name=self.config.get('region_name', 'eu-central-1'),
        )

    def extract_raw_output(self, image_path: str) -> Dict[str, Any]:
        with open(image_path, 'rb') as img:
            img_bytes = img.read()

        return self.model.detect_document_text(
            Document={'Bytes': img_bytes}
        )
