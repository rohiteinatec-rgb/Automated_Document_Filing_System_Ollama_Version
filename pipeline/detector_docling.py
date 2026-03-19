"""
detector_docling.py
-----------
GOAL: Smart extraction pipeline for PDFs.
Router: Quantrium Tech approach (Text Area vs Image Area via PyMuPDF).
Track A (Digital): Native extraction via PyMuPDF4LLM.
Track B (Scanned): Layout-aware OCR via Docling + EasyOCR.

RUN:
    python pipeline/detector_docling.py --pdf input/dummy.pdf --debug
"""
import os
import argparse
import fitz  # PyMuPDF
from pathlib import Path

import pymupdf4llm
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.datamodel.base_models import InputFormat

OUTPUT_FOLDER = "output"


def is_text_based_pdf(pdf_path, debug=False):
    """
    Quantrium Tech Approach:
    Classifies a PDF by calculating the geometric area of text blocks vs image blocks.
    """
    doc = fitz.open(pdf_path)
    page = doc[0]
    image_area = 0.0
    text_area = 0.0

    for b in page.get_text("blocks"):
        block_type = b[6]
        r = fitz.Rect(b[:4])

        if block_type == 1:
            image_area += abs(r)
        elif block_type == 0:
            text_area += abs(r)

    doc.close()

    if debug:
        print(f"[DEBUG] Geometry Check -> Text Area: {text_area:.2f} | Image Area: {image_area:.2f}")

    return text_area > image_area


def detector(pdf_path, lang_list, debug=False):
    """
    Routes the PDF to the appropriate extraction engine based on page geometry.
    """
    if debug:
        print("[DEBUG] Analyzing PDF geometry to route extraction...")

    is_digital = is_text_based_pdf(pdf_path, debug=debug)

    if is_digital:
        if debug:
            print("[DEBUG] Digital PDF detected. Routing to PyMuPDF4LLM for fast extraction...")

        return pymupdf4llm.to_markdown(pdf_path)

    else:
        if debug:
            print(f"[DEBUG] Scanned image detected. Routing to Docling + EasyOCR with languages: {lang_list}...")

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True
        pipeline_options.do_ocr = True

        pipeline_options.ocr_options = EasyOcrOptions(
            force_full_page_ocr=True,
            lang=lang_list
        )

        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        # FIXED INDENTATION HERE
        result = converter.convert(pdf_path)
        text   = result.document.export_to_markdown()

        return text


def run():
    parser = argparse.ArgumentParser(description="Extract text from a PDF and save to output folder.")
    parser.add_argument('--pdf', required=True, help='Path to the PDF file to process')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--lang', default='es+en', help='Language codes separated by a plus (e.g., es+en)')

    args = parser.parse_args()
    lang_list = args.lang.split('+')

    pdf_path = args.pdf

    if args.debug:
        print(f"[DEBUG] Reading : {pdf_path}")

    raw_text = detector(pdf_path, lang_list=lang_list, debug=args.debug)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_file = Path(OUTPUT_FOLDER) / f"{Path(pdf_path).stem}.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(raw_text)

    print(f"Characters extracted : {len(raw_text)}")
    print(f"Saved to             : {output_file}")


if __name__ == "__main__":
    run()