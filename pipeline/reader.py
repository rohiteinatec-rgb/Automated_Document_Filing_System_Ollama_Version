import pymupdf4llm
import pdfplumber
from config import Config
from quality import QualityGate

class PDFReader:
    @staticmethod
    def extract_for_classification(pdf_path: str, debug: bool = False) -> tuple:
        """Strict Digital-Only Extraction for CPU."""
        gate = QualityGate(debug)

        try:
            # Tier 1: Fast Markdown Extraction
            text = pymupdf4llm.to_markdown(pdf_path)
            eval_result = gate.evaluate(text)

            if eval_result["passed"]:
                return text[:Config.CHARS_FOR_CLASSIFICATION], "digital-markdown"

            # Tier 2: Layout-Preserving Text Fallback
            if debug: print(f"  [Reader] Tier 1 weak ({eval_result['issues']}). Trying pdfplumber...")
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join([p.extract_text() or "" for p in pdf.pages])

            return text[:Config.CHARS_FOR_CLASSIFICATION], "digital-raw"

        except Exception as e:
            if debug: print(f"  [Reader] Extraction failed: {e}")
            return "", "error"