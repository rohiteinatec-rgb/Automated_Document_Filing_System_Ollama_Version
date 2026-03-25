import pymupdf4llm
import pdfplumber
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.datamodel.base_models import InputFormat

from quality import QualityGate
from pdf_handler import PDFHandler
from vision import VisionAI

class DigitalExtractor:
    def __init__(self):
        self.quality_gate = QualityGate()

    def extract(self, pdf_path: str, debug: bool = False) -> str:
        if debug: print("  -> Running Tier 1 (PyMuPDF4LLM)")
        text = pymupdf4llm.to_markdown(pdf_path)
        if self.quality_gate.evaluate(text, debug)["passed"]: return text

        if debug: print("  -> Tier 1 failed. Escalating to Tier 2 (pdfplumber)")
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # 1. Extract regular text (layout=True helps preserve visual spacing)
                pt = page.extract_text(layout=True)
                if pt: text += pt + "\n\n"

                # 2. Extract tables and format them as Markdown
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        for row_idx, row in enumerate(table):
                            # Clean empty cells and remove line breaks within cells
                            clean_row = [str(cell).replace('\n', ' ').strip() if cell else "" for cell in row]
                            text += "| " + " | ".join(clean_row) + " |\n"

                            # Add the Markdown header separator under the first row
                            if row_idx == 0:
                                text += "|" + "|".join(["---"] * len(clean_row)) + "|\n"
                        text += "\n"
        if self.quality_gate.evaluate(text, debug)["passed"]: return text

        if debug: print("  -> Tier 2 failed. Escalating to Tier 3 (Docling Digital)")
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
        return converter.convert(pdf_path).document.export_to_markdown()


class ScannedExtractor:
    def __init__(self):
        self.quality_gate = QualityGate()
        self.vision_ai = VisionAI()

    def extract(self, pdf_path: str, lang_list: list, debug: bool = False) -> str:
        # Step 1: EasyOCR
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.ocr_options = EasyOcrOptions(force_full_page_ocr=True, lang=lang_list)
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})

        easyocr_text = converter.convert(pdf_path).document.export_to_markdown()
        qg = self.quality_gate.evaluate(easyocr_text, debug)

        if qg["passed"]:
            if debug: print("  -> EasyOCR passed quality gate. Skipping Vision AI.")
            return easyocr_text

        # Step 2: Vision AI Correction
        if debug: print(f"  -> EasyOCR failed: {qg['issues']}. Escalating to Ollama Vision AI...")
        images_b64 = PDFHandler.to_base64_images(pdf_path)

        if not images_b64:
            if debug: print("  -> Image render failed, returning base EasyOCR text.")
            return easyocr_text

        return self.vision_ai.correct_document(images_b64, easyocr_text, debug)