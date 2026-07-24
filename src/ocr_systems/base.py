from abc import ABC, abstractmethod
from typing import Any, Dict


class OCRSystem(ABC):
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config

    @abstractmethod
    def extract_raw_output(self, image_path: str) -> Dict[str, Any]:
        raise NotImplementedError
