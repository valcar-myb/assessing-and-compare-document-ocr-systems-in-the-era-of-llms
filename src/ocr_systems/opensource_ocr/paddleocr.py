"""
PaddleOCR implementation.
"""

from typing import Dict, Any
from ..base import OCRSystem


class PaddleOCROCR(OCRSystem):
    """PaddleOCR implementation."""

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.model = None
        self._init_predictor()

    def _init_predictor(self):
        """Initialize the PaddleOCR model."""
        try:
            import paddleocr
            self.model = paddleocr.PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_detection_model_name=self.config.get("det_model", "PP-OCRv3"),
                text_recognition_model_name=self.config.get("rec_model", "PP-OCRv3"),
                lang=self.config.get("lang", "en"),
            )
        except ImportError:
            print("PaddleOCR not installed. Please install with: pip install paddleocr")
            raise

    def extract_raw_output(self, image_path: str) -> Dict[str, Any]:
        """Extract raw output from PaddleOCR."""
        try:
            result = self.model.predict(image_path)
            if result and len(result) > 0:
                return result[0]._to_json().get("res", {})
            return {}
        except Exception as e:
            print(f"Error extracting raw output with PaddleOCR: {e}")
            return {}
