"""
test_vision_only.py
-------------------
GOAL: Isolate and test the raw capability of Llama 3.2 Vision.
Bypasses all routers, PyMuPDF, and EasyOCR.
Sends the image directly to Ollama and asks it to extract everything from scratch.

USAGE:
  python test_vision_only.py --pdf ../input/sc_01_automocion.pdf
"""

import os
import io
import base64
import time
import argparse
import requests
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageEnhance

# ── Configuration ─────────────────────────────────────────────
OLLAMA_BASE_URL  = "http://localhost:11434"
OLLAMA_MODEL     = "llama3.2-vision:latest"
OLLAMA_TIMEOUT   = 400  # 15 minutes (Protects against CPU timeout)
RENDER_DPI       = 300  # Lowered from 300 to drastically speed up CPU inference
OUTPUT_FOLDER    = "output_vision_test"

def pdf_to_base64_images(pdf_path: str) -> list:
    """Renders PDF pages to base64 images to feed directly to Ollama."""
    images_b64 = []
    print(f"[*] Rendering {pdf_path} to images at {RENDER_DPI} DPI...")
    try:
        doc = fitz.open(pdf_path)
        mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Enhance contrast to help the AI read faded text
            img = ImageEnhance.Sharpness(img).enhance(2.0)
            img = ImageEnhance.Contrast(img).enhance(1.8)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            images_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        doc.close()
    except Exception as e:
        print(f"[!] Image rendering failed: {e}")
    return images_b64

def run_vision_test(pdf_path: str):
    images_b64 = pdf_to_base64_images(pdf_path)

    if not images_b64:
        print("[!] No images generated. Exiting.")
        return

    extracted_pages = []
    print(f"[*] Sending {len(images_b64)} page(s) to {OLLAMA_MODEL}...")

    for i, img_b64 in enumerate(images_b64):
        # Pure Zero-Shot Prompt (No OCR baseline provided)
        prompt = """You are a highly precise data extraction AI. Read the attached Spanish invoice image.
        
        CRITICAL INSTRUCTIONS:
        1. Extract EVERY piece of text visible on the page.
        2. Format the output perfectly using Markdown.
        3. Reconstruct all financial tables perfectly using Markdown table syntax (| Column | Column |).
        4. Do not miss the footer at the bottom of the page (Look carefully for the IBAN, Registro Mercantil, and addresses).
        5. Output ONLY the extracted Markdown text. Do not add conversational greetings."""

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {
                "temperature": 0.0, # Zero creativity, maximum precision
                "num_predict": 1500
            }
        }

        try:
            start_time = time.time()
            print(f"    -> Processing page {i+1} (Please wait, this relies on your CPU...)")

            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=OLLAMA_TIMEOUT
            )
            resp.raise_for_status()

            elapsed = time.time() - start_time
            result_text = resp.json().get("response", "").strip()
            print(f"    -> Page {i+1} completed in {elapsed:.1f} seconds! ({len(result_text)} characters)")

            extracted_pages.append(result_text)

        except requests.exceptions.Timeout:
            print(f"    [!] Vision AI timed out after {OLLAMA_TIMEOUT} seconds on page {i+1}.")
        except Exception as e:
            print(f"    [!] API Error on page {i+1}: {e}")

    # Save the output
    if extracted_pages:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        out_file = Path(OUTPUT_FOLDER) / f"{Path(pdf_path).stem}_vision_only.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n\n---\n\n".join(extracted_pages))
        print(f"\n[SUCCESS] Raw Vision AI extraction saved to: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', required=True, help="Path to the PDF to test")
    args = parser.parse_args()

    run_vision_test(args.pdf)