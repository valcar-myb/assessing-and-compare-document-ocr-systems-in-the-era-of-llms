"""
Microsoft Azure AI Vision OCR implementation.
"""

from typing import Dict, Any
from ..base import OCRSystem


class AzureVisionOCR(OCRSystem):
    """Microsoft Azure AI Vision OCR implementation."""

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.model = None
        self._init_predictor()

    def _init_predictor(self):
        """Initialize the Azure Vision client."""
        try:
            from azure.ai.vision.imageanalysis import ImageAnalysisClient
            from azure.ai.vision.imageanalysis.models import VisualFeatures
            from azure.core.credentials import AzureKeyCredential

            self.visual_features = VisualFeatures

            self.model = ImageAnalysisClient(
                endpoint=self.config.get('endpoint'),
                credential=AzureKeyCredential(key=self.config.get('credential')),
            )
        except ImportError:
            print("Azure AI Vision SDK not installed. Please install with: pip install azure-ai-vision-imageanalysis")
            raise
        except Exception as e:
            print(f"Error initializing Azure Vision client: {e}")
            raise

    def extract_raw_output(self, image_path: str) -> Dict[str, Any]:
        """Extract raw output from Azure Vision."""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()

            result = self.model.analyze(
                image_data=image_data,
                visual_features=[self.visual_features.READ],
                model_version=self.config.get('model_version', 'latest'),
            )
            return result.as_dict()
        except Exception as e:
            print(f"Error extracting raw output with Azure Vision: {e}")
            return {}
