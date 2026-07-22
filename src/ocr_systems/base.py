"""Base interface for OCR / document-transcription systems."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class OCRSystem(ABC):
    """
    Abstract base class for an OCR system.

    Subclasses implement `extract_raw_output`, which runs the system on a single
    image and returns its raw output (as returned by the engine/API). Converting
    that raw output into plain text is handled by the parsers in
    `src/parsing/parsers.py`.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config

    @abstractmethod
    def extract_raw_output(self, image_path: str) -> Dict[str, Any]:
        """Run the system on a single image and return its raw output."""
        raise NotImplementedError
