"""
DocTR OCR implementation.
"""

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import torch
from typing import Dict, Any
from ..base import OCRSystem


class DocTROCR(OCRSystem):
    """DocTR OCR system implementation."""

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.det_arch = config.get('det_arch', 'db_resnet50')
        self.reco_arch = config.get('reco_arch', 'crnn_vgg16_bn')
        self.pretrained = config.get('pretrained', True)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        self.predictor = ocr_predictor(
            det_arch=self.det_arch,
            reco_arch=self.reco_arch,
            pretrained=self.pretrained,
        ).to(self.device)

    def extract_raw_output(self, image_path: str) -> Dict[str, Any]:
        """Extract raw DocTR output from an image."""
        doc = DocumentFile.from_images(image_path)
        result = self.predictor(doc)
        return result.export()
