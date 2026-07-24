from typing import Dict, Any

import paddleocr

from ..base import OCRSystem


class PaddleOCROCR(OCRSystem):
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.model = paddleocr.PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_detection_model_name=self.config.get("det_model", "PP-OCRv3"),
            text_recognition_model_name=self.config.get("rec_model", "PP-OCRv3"),
            lang=self.config.get("lang", "en"),
        )

    def extract_raw_output(self, image_path: str) -> Dict[str, Any]:
        result = self.model.predict(image_path)
        if result and len(result) > 0:
            return result[0]._to_json().get("res", {})
        return {}
