"""
Project 2 — Document Auto-Filer
reader.py — extract text from PDF for classification

95% of documents are digital. pymuPDF extracts text in milliseconds.
No AI, no OCR, no GPU needed for this step.

For the rare scanned PDF: falls back to a simple image-based extraction
hint so the classifier still gets something to work with.
"""
import fitz   # pymuPDF
from config import Config


class PDFReader:
    """
    Extracts the first N characters from a PDF for classification.
    We only need enough to identify the document type — not full extraction.
    """

    @staticmethod
    def extract_for_classification(pdf_path: str,
                                   debug: bool = False) -> tuple:
        """
        Extract text from PDF for classification purposes.

        Returns:
            (text: str, method: str)
            text   — extracted text, up to CHARS_FOR_CLASSIFICATION chars
            method — how it was extracted: "digital" or "scanned_hint"
        """
        try:
            doc          = fitz.open(pdf_path)
            full_text    = ""
            total_chars  = 0

            for page in doc:
                page_text = page.get_text().strip()
                full_text += page_text + "\n"
                total_chars += len(page_text)
                # Stop once we have enough — no need to read full document
                if total_chars >= Config.CHARS_FOR_CLASSIFICATION:
                    break

            doc.close()
            text = full_text.strip()[:Config.CHARS_FOR_CLASSIFICATION]

            if len(text) >= 50:
                if debug:
                    print(f"  [Reader] Digital path — {len(text)} chars extracted")
                return text, "digital"

            # Very little text found — likely scanned
            # Return a minimal hint so classifier can still try
            if debug:
                print(f"  [Reader] Only {len(text)} chars found — likely scanned PDF")
                print(f"  [Reader] Returning available text as hint")
            return text, "scanned_hint"

        except Exception as e:
            if debug:
                print(f"  [Reader] Error reading PDF: {e}")
            return "", "error"

    @staticmethod
    def is_readable(pdf_path: str) -> bool:
        """Quick check — can we open and read this PDF at all?"""
        try:
            doc   = fitz.open(pdf_path)
            pages = len(doc)
            doc.close()
            return pages > 0
        except Exception:
            return False
