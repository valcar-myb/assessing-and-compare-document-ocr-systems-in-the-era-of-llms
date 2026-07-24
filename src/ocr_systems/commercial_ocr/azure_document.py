from typing import Dict, Any

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

from ..base import OCRSystem


class AzureDocumentOCR(OCRSystem):
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.model = DocumentIntelligenceClient(
            endpoint=self.config.get('endpoint'),
            credential=AzureKeyCredential(key=self.config.get('credential')),
        )

    def extract_raw_output(self, image_path: str) -> Dict[str, Any]:
        with open(image_path, 'rb') as image_file:
            content = image_file.read()

        poller = self.model.begin_analyze_document(
            model_id=self.config.get('model_id', 'prebuilt-read'),
            body=content,
        )
        return poller.result().as_dict()
