"""
Tesseract OCR implementation.
"""

import pytesseract
from PIL import Image
from typing import Dict, Any
from ..base import OCRSystem


class TesseractOCR(OCRSystem):
    """Tesseract OCR system implementation."""

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.language = config.get('language', 'eng')
        self.psm = config.get('psm', 3)
        self.oem = config.get('oem', 3)

    def extract_raw_output(self, image_path: str) -> Dict[str, Any]:
        """Extract raw Tesseract output from an image."""
        image = Image.open(image_path)

        data = pytesseract.image_to_data(
            image,
            lang=self.language,
            config=f'--psm {self.psm} --oem {self.oem}',
            output_type=pytesseract.Output.DICT,
        )

        text = pytesseract.image_to_string(
            image,
            lang=self.language,
            config=f'--psm {self.psm} --oem {self.oem}',
        )

        return {
            'text': text.strip(),
            'data': data,
            'language': self.language,
            'psm': self.psm,
            'oem': self.oem,
        }
