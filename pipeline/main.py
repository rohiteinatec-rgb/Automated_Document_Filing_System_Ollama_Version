import os
import argparse
from pathlib import Path

from config import Config
from pdf_handler import PDFHandler
from extractors import DigitalExtractor, ScannedExtractor

class PipelineOrchestrator:
    def process(self, pdf_path: str, lang_list: list, debug: bool = False) -> str:
        if debug: print("[DEBUG] Analyzing PDF geometry to route extraction...")

        if PDFHandler.is_text_based(pdf_path):
            if debug: print("\n[DEBUG] Digital PDF detected. Starting Track A...")
            extractor = DigitalExtractor()
            return extractor.extract(pdf_path, debug)
        else:
            if debug: print(f"\n[DEBUG] Scanned image detected. Starting Track B (langs: {lang_list})...")
            extractor = ScannedExtractor()
            return extractor.extract(pdf_path, lang_list, debug)

def run():
    parser = argparse.ArgumentParser(description="Modular PDF Extraction Pipeline")
    parser.add_argument('--pdf', required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--lang', default='+'.join(Config.DEFAULT_LANGS))
    args = parser.parse_args()

    os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)
    out_file = Path(Config.OUTPUT_FOLDER) / f"{Path(args.pdf).stem}.txt"

    orchestrator = PipelineOrchestrator()
    raw_text = orchestrator.process(args.pdf, args.lang.split('+'), debug=args.debug)

    if raw_text:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(raw_text)
        print(f"\n[SUCCESS] Extracted {len(raw_text)} chars. Saved to {out_file}")

if __name__ == "__main__":
    run()