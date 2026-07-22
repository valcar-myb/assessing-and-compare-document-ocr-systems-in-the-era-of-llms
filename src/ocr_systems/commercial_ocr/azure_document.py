"""
Microsoft Azure Document Intelligence OCR implementation.
"""

from typing import Dict, Any
from ..base import OCRSystem


class AzureDocumentOCR(OCRSystem):
    """Microsoft Azure Document Intelligence OCR implementation."""

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.model = None
        self._init_predictor()

    def _init_predictor(self):
        """Initialize the Azure Document Intelligence client."""
        try:
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            from azure.core.credentials import AzureKeyCredential

            self.model = DocumentIntelligenceClient(
                endpoint=self.config.get('endpoint'),
                credential=AzureKeyCredential(key=self.config.get('credential')),
            )
        except ImportError:
            print("Azure Document Intelligence SDK not installed. Please install with: pip install azure-ai-documentintelligence")
            raise
        except Exception as e:
            print(f"Error initializing Azure Document Intelligence client: {e}")
            raise

    def extract_raw_output(self, image_path: str) -> Dict[str, Any]:
        """Extract raw output from Azure Document Intelligence."""
        try:
            with open(image_path, 'rb') as image_file:
                content = image_file.read()

            poller = self.model.begin_analyze_document(
                model_id=self.config.get('model_id', 'prebuilt-read'),
                body=content,
            )
            result = poller.result()
            return result.as_dict()
        except Exception as e:
            print(f"Error extracting raw output with Azure Document Intelligence: {e}")
            return {}
