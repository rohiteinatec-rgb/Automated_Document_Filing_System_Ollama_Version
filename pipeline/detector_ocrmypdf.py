"""
detector_ocr.py
---------------
GOAL: Read one scanned invoice PDF → save raw text to /output

Output : Extract text from a scanned PDF and save to /output

RUN:
    python pipeline/detector_ocr.py --pdf input/scanned_invoice_test.pdf
    python pipeline/detector_ocr.py --pdf input/dummy.pdf
"""

import os
import argparse
import subprocess
import fitz
from pathlib import Path

OUTPUT_FOLDER = "output"

def is_text_based_pdf(pdf_path, debug=False):
    """
    Classifies a PDF as text-based or image-based by calculating the
    geometric area of text blocks vs image blocks (Based on Quantrium Tech approach).
    """
    doc = fitz.open(pdf_path)
    page = doc[0]  # Checking the first page

    image_area = 0.0
    text_area = 0.0

    # get_text("blocks") returns a list of blocks.
    # Each block is a tuple: (x0, y0, x1, y1, text/image details, block_no, block_type)
    for b in page.get_text("blocks"):
        block_type = b[6] # 0 for text, 1 for image

        if block_type == 1:
            r = fitz.Rect(b[:4])
            image_area += abs(r)
        elif block_type == 0:
            r = fitz.Rect(b[:4])
            text_area += abs(r)

    doc.close()

    if debug:
        print(f"[DEBUG] Geometry Check -> Text Area: {text_area:.2f} | Image Area: {image_area:.2f}")

    # Classification Logic
    if text_area == 0.0 and image_area != 0.0:
        return False  # Pure image-based scan
    elif image_area == 0.0 and text_area != 0.0:
        return True   # Pure text-based digital PDF
    else:
        # Mixed document
        return text_area > image_area

def extract_native_text(pdf_path):
    """Reads digital text instantly."""
    doc = fitz.open(pdf_path)
    text = doc[0].get_text()
    doc.close()
    return text.strip()


def smart_detector(pdf_path, lang_code, debug=False):
    """
    ROUTER LOGIC: Detects if the PDF is digital or scanned using geometric block analysis.
    """
    if debug:
        print("[DEBUG] Analyzing PDF geometry to determine extraction strategy...")

    # 1. Use the area-based logic to classify the PDF
    is_digital = is_text_based_pdf(pdf_path, debug)

    # 2. Route based on the classification
    if is_digital:
        if debug:
            print("[DEBUG] Text dominates the page. Digital PDF detected. Skipping Docker OCR.")
        return extract_native_text(pdf_path)
    else:
        if debug:
            print("[DEBUG] Images dominate the page. Scanned PDF detected. Booting Docker OCR...")
        return detector(pdf_path, lang_code)

def detector(pdf_path, lang_code):
    """
    Spawns a Docker child process to run OCRmyPDF.
    Reads the text, cleans up the Docker temporary files, and returns the string.

    Scanned PDF → returns full invoice text via OCR
    """
    abs_input_path = Path(pdf_path).resolve()
    input_dir = abs_input_path.parent

    # We use the output folder for Docker's temporary working files
    temp_dir = Path(OUTPUT_FOLDER).resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_pdf = temp_dir / "temp_ocr.pdf"
    temp_txt = temp_dir / "temp_sidecar.txt"

    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{input_dir}:/input",
        "-v", f"{temp_dir}:/output",
        "jbarlow83/ocrmypdf",
        "--force-ocr",
        "-l", lang_code,
        "--tesseract-config", "pagesegmode=6", # Try 6 (Assume uniform block) or 4 (Assume single column of text of variable sizes)
        "--sidecar", f"/output/{temp_txt.name}",
        f"/input/{abs_input_path.name}",
        f"/output/{temp_pdf.name}"
    ]

    try:
        # Run the Docker container
        subprocess.run(docker_cmd, capture_output=True, text=True, check=True)

        # Read the text extracted by Docker
        with open(temp_txt, "r", encoding="utf-8") as f:
            text = f.read()

        return text

    finally:
        # Cleanup: Delete Docker's temporary files before returning the string
        if temp_pdf.exists(): temp_pdf.unlink()
        if temp_txt.exists(): temp_txt.unlink()


def run():
    parser = argparse.ArgumentParser(description="Extract text from a PDF and save to output folder.")
    parser.add_argument('--pdf', required=True, help='Path to the PDF file to process')
    parser.add_argument('--lang', default='eng+spa+cat', help='Language codes (e.g., spa, eng+spa). Default: eng+spa+cat')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()

    pdf_path = args.pdf


    if args.debug:
        print(f"[DEBUG] Reading : {pdf_path}")

    try:
        # 1. Get the raw text (this now calls the Docker OCR pipeline)
        raw_text = smart_detector(pdf_path, args.lang, args.debug)

        # 2. Save to /output using the exact same filename as the PDF
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        output_file = Path(OUTPUT_FOLDER) / f"{Path(pdf_path).stem}.txt"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(raw_text)

        print(f"Characters extracted : {len(raw_text)}")
        print(f"Saved to             : {output_file}")

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] OCR Engine Failed.\nDocker Log:\n{e.stderr}")
    except Exception as e:
        print(f"[ERROR] Pipeline stopped: {e}")


if __name__ == "__main__":
    run()