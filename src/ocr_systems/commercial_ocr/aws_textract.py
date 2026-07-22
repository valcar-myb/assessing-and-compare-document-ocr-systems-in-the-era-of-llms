"""
AWS Textract OCR implementation.
"""

import boto3
from typing import Dict, Any
from ..base import OCRSystem


class AWSTextractOCR(OCRSystem):
    """AWS Textract OCR implementation."""

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.model = None
        self._init_predictor()

    def _init_predictor(self):
        """Initialize the AWS Textract client."""
        try:
            self.model = boto3.client(
                service_name='textract',
                aws_access_key_id=self.config.get('aws_access_key_id'),
                aws_secret_access_key=self.config.get('aws_secret_access_key'),
                region_name=self.config.get('region_name', 'eu-central-1'),
            )
        except Exception as e:
            print(f"Error initializing AWS Textract client: {e}")
            raise

    def extract_raw_output(self, image_path: str) -> Dict[str, Any]:
        """Extract raw output from AWS Textract."""
        try:
            with open(image_path, 'rb') as img:
                img_bytes = img.read()

            response = self.model.detect_document_text(
                Document={'Bytes': img_bytes}
            )
            return response
        except Exception as e:
            print(f"Error extracting raw output with AWS Textract: {e}")
            return {}
