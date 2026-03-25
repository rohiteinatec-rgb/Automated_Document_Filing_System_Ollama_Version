"""
detector_ollama.py
------------------
GOAL: Advanced Extraction Pipeline with Self-Healing Vision AI (Spain-Optimized).
Track A (Digital): Native extraction (PyMuPDF4LLM -> pdfplumber -> Docling)
Track B (Scanned): EasyOCR -> Ollama Vision AI Correction (llama3.2-vision)

USAGE:
  python detector_ollama.py --pdf invoice.pdf --debug
"""

import os
import re
import io
import base64
import time
import argparse
import sys
import requests
from pathlib import Path

import fitz  # PyMuPDF
import pymupdf4llm
import pdfplumber
from PIL import Image, ImageEnhance

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.datamodel.base_models import InputFormat

# ── Configuration ─────────────────────────────────────────────
OLLAMA_BASE_URL  = "http://localhost:11434"
OLLAMA_MODEL     = "llama3.2-vision:latest"
OLLAMA_TIMEOUT   = 300
RENDER_DPI       = 300
OUTPUT_FOLDER    = "output"
DEFAULT_LANGS    = ["es", "en"]

# Spain-Specific Invoice Signals
INVOICE_FIELD_PATTERNS = {
    "cif_nie":    re.compile(r'\b[A-Z]\d{7}[A-Z0-9]\b'),
    "iban":       re.compile(r'\bES\d{2}[\s\d]{20,26}\b', re.IGNORECASE),
    "euro_amount":re.compile(r'\d{1,3}(?:[.,]\d{3})*[.,]\d{2}\s*€'),
    "date":       re.compile(r'\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b'),
    "invoice_num":re.compile(r'\b[A-Z]{2,6}[/\-]\d{4}[/\-]\d{3,6}\b'),
}
MIN_FIELDS_REQUIRED = 2

# ════════════════════════════════════════════════════════════
# CORE UTILITIES (QUALITY GATE & ROUTER)
# ════════════════════════════════════════════════════════════

def quality_gate(text: str, debug: bool = False) -> dict:
    """Two-layer quality check: Generic health, Field-level semantics, and Hallucination detection."""
    issues       = []
    fields_found = []
    score        = 100

    if not text or not text.strip():
        return {"passed": False, "score": 0, "issues": ["empty output"], "fields_found": []}

    stripped = text.strip()

    # ── Layer 1: Generic ──────────────────────────────────────
    if len(stripped) < 80:
        issues.append(f"suspiciously short: {len(stripped)} chars")
        score -= 40
    if "(cid:" in stripped:
        issues.append(f"font encoding corruption: {stripped.count('(cid:')} sequences")
        score -= 50

    printable   = sum(1 for c in stripped if c.isprintable() or c in "\n\t")
    garbage_pct = 1 - (printable / max(len(stripped), 1))
    if garbage_pct > 0.15:
        issues.append(f"high garbage ratio: {garbage_pct:.1%}")
        score -= 30

    # ── Layer 2: Hallucination & Corruption Detection (NEW) ───
    # Catch 1: Letters mistakenly read inside numbers (e.g., "6.7oo.00")
    if re.search(r'\d+[a-zA-Z]+\d+', stripped):
        issues.append("OCR number corruption detected (letters inside numbers)")
        score -= 50

    # Catch 2: Mangled percentages (EasyOCR often reads '%' as '86' or '96')
    if re.search(r'\(\d{2}86\)', stripped) or re.search(r'\(\d{2}96\)', stripped):
        issues.append("OCR percentage corruption detected (misread % symbol)")
        score -= 40

    # Catch 3: Empty cells next to critical financial keywords
    # This catches: "| IVA (21%) |    |" where the amount is completely dropped
    if re.search(r'\|\s*(?:IVA|TOTAL|Base Imponible)[^\|]*\|\s*\|', stripped, re.IGNORECASE):
        issues.append("Critical financial table value is missing/empty")
        score -= 40

    # ── Layer 3: Field-level Semantics ────────────────────────
    for field_name, pattern in INVOICE_FIELD_PATTERNS.items():
        if pattern.search(stripped):
            fields_found.append(field_name)

    if len(fields_found) < MIN_FIELDS_REQUIRED:
        issues.append(f"too few fields detected: {len(fields_found)}/{len(INVOICE_FIELD_PATTERNS)}")
        score -= 25

    # NEW: Strict IBAN/Bank detail enforcement
    if "iban" not in fields_found:
        issues.append("No IBAN or Bank Details found (likely missed footer)")
        score -= 30 # Heavy penalty to force Vision AI escalation

    # Final Evaluation
    score  = max(0, score)
    passed = score >= 80 and len(fields_found) >= MIN_FIELDS_REQUIRED # Raised passing bar from 60 to 80

    if debug:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"    [QualityGate] {status} — score={score} | fields={fields_found} | issues={issues}")

    return {"passed": passed, "score": score, "issues": issues, "fields_found": fields_found}


def is_text_based_pdf(pdf_path: str, debug: bool = False) -> bool:
    """The Gatekeeper: Detects digital vs scanned PDFs via geometry."""
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]
        text_area, image_area = 0.0, 0.0

        for block in page.get_text("blocks"):
            area = abs(fitz.Rect(block[:4]))
            if block[6] == 0: text_area += area
            elif block[6] == 1: image_area += area
        doc.close()

        return text_area > image_area
    except Exception:
        return False

# ════════════════════════════════════════════════════════════
# VISION AI UTILITIES
# ════════════════════════════════════════════════════════════

def _pdf_to_images(pdf_path: str, dpi: int = RENDER_DPI) -> list:
    """Renders PDF pages to enhanced base64 images for Ollama."""
    images_b64 = []
    try:
        doc = fitz.open(pdf_path)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Enhance for better Vision AI reading
            img = ImageEnhance.Sharpness(img).enhance(2.0)
            img = ImageEnhance.Contrast(img).enhance(1.8)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            images_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        doc.close()
    except Exception as e:
        print(f"    [ImageRender] Failed: {e}")
    return images_b64


def _remove_loops(text: str, max_chars: int = 1800) -> str:
    """Removes hallucination repetition loops from local Vision AI output."""
    if not text: return text
    if len(text) > max_chars: text = text[:text.rfind('\n', 0, max_chars)]

    seen, result = {}, []
    for line in text.splitlines():
        key = line.strip().lower()
        if len(key) < 15:
            result.append(line)
            continue
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= 2: result.append(line)
        else: break
    return "\n".join(result).strip()

# ════════════════════════════════════════════════════════════
# ROUTER & PIPELINES
# ════════════════════════════════════════════════════════════

def detector(pdf_path: str, lang_list: list, debug: bool = False) -> str:
    """Routes the PDF and executes the appropriate pipeline."""
    if debug: print("[DEBUG] Analyzing PDF geometry to route extraction...")

    is_digital = is_text_based_pdf(pdf_path, debug=debug)

    if is_digital:
        # ==========================================
        # TRACK A: DIGITAL PIPELINE
        # ==========================================
        if debug: print("\n[DEBUG] Digital PDF detected. Starting Track A...")

        if debug: print("  -> Running Tier 1 (PyMuPDF4LLM)")
        text = pymupdf4llm.to_markdown(pdf_path)
        qg = quality_gate(text, debug)

        if not qg["passed"]:
            if debug: print("  -> Tier 1 failed quality gate. Escalating to Tier 2 (pdfplumber)")
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    pt = page.extract_text()
                    if pt: text += pt + "\n"
            qg = quality_gate(text, debug)

            if not qg["passed"]:
                if debug: print("  -> Tier 2 failed. Escalating to Tier 3 (Docling Digital)")
                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = False
                pipeline_options.do_table_structure = True
                converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
                text = converter.convert(pdf_path).document.export_to_markdown()

        return text

    else:
        # ==========================================
        # TRACK B: SCANNED PIPELINE WITH VISION AI
        # ==========================================
        if debug: print(f"\n[DEBUG] Scanned image detected. Starting Track B (langs: {lang_list})...")

        # Step 1: EasyOCR
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.ocr_options = EasyOcrOptions(force_full_page_ocr=True, lang=lang_list)
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})

        easyocr_text = converter.convert(pdf_path).document.export_to_markdown()
        qg = quality_gate(easyocr_text, debug)

        if qg["passed"]:
            if debug: print("  -> EasyOCR passed quality gate. Skipping Vision AI.")
            return easyocr_text

        # Step 2: Vision AI Correction
        if debug: print(f"  -> EasyOCR failed: {qg['issues']}. Escalating to Ollama Vision AI...")
        images_b64 = _pdf_to_images(pdf_path)

        if not images_b64:
            if debug: print("  -> Image render failed, returning base EasyOCR text.")
            return easyocr_text

        corrected_pages = []
        for i, img_b64 in enumerate(images_b64):
            context = easyocr_text[:1200]

            # Spain-Optimized "Anti-Lazy" Prompt
            prompt = f"""You are a precise data extraction AI analyzing a Spanish invoice. Compare the attached image with this flawed OCR text:
            
            --- OCR TEXT ---
            {context}
            --- END OCR TEXT ---
            
            Produce a flawlessly CORRECTED and COMPLETE Markdown version of the invoice. 
            
            CRITICAL INSTRUCTIONS:
            1. FIX CORRUPTED NUMBERS: Correct any letters mixed into numbers (e.g., "6.7oo.00" -> "6.700,00").
            2. FIX PERCENTAGES: Correct misread symbols (e.g., "IVA (2186)" -> "IVA (21%)").
            3. REBUILD THE TABLE: Ensure no financial values are left blank if they exist in the image.
            4. EXTRACT THE FOOTER (MANDATORY): Look at the very bottom margins of the image. You MUST extract all bank details, the IBAN (starting with ES), and company addresses. Do not skip the small print.
            5. Preserve Spanish terminology exactly (CIF, NIF, IVA, Base Imponible).
            6. Output ONLY the corrected Markdown. No conversational text.
            
            Output the corrected Markdown now:"""

            try:
                if debug: print(f"    [VisionAI] Correcting page {i+1}...")
                resp = requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={"model": OLLAMA_MODEL, "prompt": prompt, "images": [img_b64], "stream": False, "options": {"temperature": 0.01}},
                    timeout=OLLAMA_TIMEOUT
                )
                resp.raise_for_status()
                corrected = _remove_loops(resp.json().get("response", "").strip())
                corrected_pages.append(corrected)
            except Exception as e:
                if debug: print(f"    [VisionAI] Error on page {i+1}: {e}")
                corrected_pages.append(easyocr_text)

        return "\n\n".join(corrected_pages)


def run():
    parser = argparse.ArgumentParser(description="Spain-Optimized PDF Extractor with Vision AI")
    parser.add_argument('--pdf', required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--lang', default='es+en')
    args = parser.parse_args()

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    out_file = Path(OUTPUT_FOLDER) / f"{Path(args.pdf).stem}.txt"

    raw_text = detector(args.pdf, args.lang.split('+'), debug=args.debug)

    if raw_text:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(raw_text)
        print(f"\n[SUCCESS] Extracted {len(raw_text)} chars. Saved to {out_file}")

if __name__ == "__main__":
    run()