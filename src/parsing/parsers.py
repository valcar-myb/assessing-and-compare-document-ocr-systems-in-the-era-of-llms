"""
Output parsers - one method per evaluated system.

Each parser turns a system's raw output into plain text. Line breaks are
**preserved**: line-level engines join their lines with ``\\n`` and free-text
LLM outputs keep their original newlines. This matters for Flexible Character
Accuracy (see docs/metrics.md), which derives line units from ``\\n``. The
line-insensitive metrics (CA, WA, NED) are unaffected because ocreval and the
edit-distance metrics treat newlines as ordinary whitespace.
"""

from typing import Dict, Any


class OCRParser:
    """Parser for OCR / LLM raw outputs."""

    @staticmethod
    def parse_doctr(raw_data: Dict[str, Any]) -> str:
        """Parse DocTR raw output and return extracted text."""
        text_lines = []
        for page in raw_data.get('pages', []):
            for block in page.get('blocks', []):
                for line in block.get('lines', []):
                    words = []
                    for word in line.get('words', []):
                        if isinstance(word, dict) and 'value' in word:
                            words.append(word['value'])
                    if words:
                        text_lines.append(' '.join(words))
        return '\n'.join(text_lines)

    @staticmethod
    def parse_paddleocr(raw_data: Dict[str, Any]) -> str:
        """Parse PaddleOCR raw output and return extracted text."""
        rec_texts = raw_data.get('rec_texts', [])
        if rec_texts:
            return '\n'.join(rec_texts)
        return ""

    @staticmethod
    def parse_tesseract(raw_data: Dict[str, Any]) -> str:
        """Parse Tesseract raw output and return extracted text."""
        text = raw_data.get('text', '')
        # Keep line breaks; only trim trailing whitespace on each line.
        lines = [line.rstrip() for line in text.splitlines()]
        return '\n'.join(lines).strip()

    @staticmethod
    def parse_aws_textract(raw_data: Dict[str, Any]) -> str:
        """Parse AWS Textract raw output and return extracted text."""
        blocks = raw_data.get('Blocks', [])
        lines = [block['Text'] for block in blocks if block.get('BlockType') == 'LINE']
        return '\n'.join(lines)

    @staticmethod
    def parse_azure_vision(raw_data: Dict[str, Any]) -> str:
        """Parse Azure Vision raw output and return extracted text."""
        blocks = raw_data.get('readResult', {}).get('blocks', [])
        lines = [line['text'] for block in blocks for line in block.get('lines', [])]
        return '\n'.join(lines)

    @staticmethod
    def parse_azure_document(raw_data: Dict[str, Any]) -> str:
        """Parse Azure Document Intelligence raw output and return extracted text."""
        pages = raw_data.get('pages', [])
        lines = [line['content'] for page in pages for line in page.get('lines', [])]
        return '\n'.join(lines)

    @staticmethod
    def parse_google_vision(raw_data: Dict[str, Any]) -> str:
        """Parse Google Vision raw output and return extracted text."""
        return raw_data.get('fullTextAnnotation', {}).get('text', '')

    @staticmethod
    def parse_google_document(raw_data: Dict[str, Any]) -> str:
        """Parse Google Document AI raw output and return extracted text."""
        return raw_data.get('document', {}).get('text', '')

    @staticmethod
    def parse_gpt4o(raw_data: Dict[str, Any]) -> str:
        """Parse GPT-4o vision raw output and return extracted text."""
        contents = [
            choice['message']['content']
            for choice in raw_data.get('choices', [])
            if 'message' in choice
        ]
        return '\n'.join(contents)

    @staticmethod
    def parse_gemini_flash(raw_data: Dict[str, Any]) -> str:
        """Parse Gemini Flash raw output and return extracted text."""
        candidates = raw_data.get('candidates', [])
        parts = [
            part['text']
            for candidate in candidates
            for part in candidate.get('content', {}).get('parts', [])
        ]
        return '\n'.join(parts)

    @staticmethod
    def parse_claude_haiku(raw_data: Dict[str, Any]) -> str:
        """Parse Claude Haiku raw output and return extracted text."""
        contents = [content['text'] for content in raw_data.get('content', [])]
        return '\n'.join(contents)

    @staticmethod
    def parse_mistral_ocr(raw_data: Dict[str, Any]) -> str:
        """Parse Mistral Document AI raw output and return extracted text."""
        pages = raw_data.get('pages', [])
        markdowns = [page['markdown'] for page in pages]
        return '\n'.join(markdowns)

    @staticmethod
    def parse_vllm_openai(raw_data: Dict[str, Any]) -> str:
        """Parse a vLLM OpenAI-compatible raw output and return extracted text."""
        contents = [
            choice['message']['content']
            for choice in raw_data.get('choices', [])
            if 'message' in choice
        ]
        return '\n'.join(contents)

    @staticmethod
    def get_parser(system_name: str):
        """Get the parser function for a specific system."""
        parsers = {
            'doctr': OCRParser.parse_doctr,
            'paddleocr': OCRParser.parse_paddleocr,
            'tesseract': OCRParser.parse_tesseract,
            'aws_textract': OCRParser.parse_aws_textract,
            'azure_vision': OCRParser.parse_azure_vision,
            'azure_document': OCRParser.parse_azure_document,
            'google_vision': OCRParser.parse_google_vision,
            'google_document': OCRParser.parse_google_document,
            'gpt4o': OCRParser.parse_gpt4o,
            'gemini_flash': OCRParser.parse_gemini_flash,
            'claude_haiku': OCRParser.parse_claude_haiku,
            'mistral_ocr': OCRParser.parse_mistral_ocr,
            # Open-source VLMs served via vLLM use the OpenAI-compatible format.
            'qwen25vl': OCRParser.parse_vllm_openai,
            'gemma3': OCRParser.parse_vllm_openai,
            'paddleocr_vl': OCRParser.parse_vllm_openai,
            'deepseek_ocr': OCRParser.parse_vllm_openai,
        }
        return parsers.get(system_name, lambda x: "")
