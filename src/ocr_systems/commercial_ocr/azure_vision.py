from typing import Dict, Any

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

from ..base import OCRSystem


class AzureVisionOCR(OCRSystem):
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.model = ImageAnalysisClient(
            endpoint=self.config.get('endpoint'),
            credential=AzureKeyCredential(key=self.config.get('credential')),
        )

    def extract_raw_output(self, image_path: str) -> Dict[str, Any]:
        with open(image_path, 'rb') as f:
            image_data = f.read()

        result = self.model.analyze(
            image_data=image_data,
            visual_features=[VisualFeatures.READ],
            model_version=self.config.get('model_version', 'latest'),
        )
        return result.as_dict()
