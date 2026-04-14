"""
GESDOC Document Classifier
main.py — orchestrator and CLI entry point

USAGE:
  # Single PDF
  python main.py --pdf ..\\input\\invoice.pdf

  # Single PDF with debug output
  python main.py --pdf ..\\input\\invoice.pdf --debug

  # Process folder — all PDFs
  python main.py --folder ..\\input
"""

import os
import sys
import time
import argparse
from pathlib import Path

from config import Config
from reader import PDFReader
from classifier import Classifier
from filer import Filer

class DocumentAutoFiler:
    def __init__(self, debug: bool = False, dry_run: bool = False):
        self.debug = debug
        self.dry_run = dry_run
        self.classifier = Classifier(debug)
        self.filer = Filer(debug)

    def process(self, pdf_path: str) -> dict:
        t_total_start = time.time()
        pdf_path = str(pdf_path)
        fname = Path(pdf_path).name

        print(f"\n{'─'*55}")
        print(f"  📄 {fname}")
        print(f"{'─'*55}")

        # -----------------------------------------
        # Step 1: Read (Extraction)
        # -----------------------------------------
        t_ext_start = time.time()
        text, method = PDFReader.extract_for_classification(pdf_path, self.debug)
        t_ext_ms = (time.time() - t_ext_start) * 1000

        if not text or len(text.strip()) < 20:
            print(f"  ❌ Extraction failed (0 chars) — skipping")
            return {"success": False, "file": fname, "action": "error", "message": "unreadable PDF"}

        if self.debug:
            print(f"  [Read] Method={method} | {len(text)} chars extracted")

        # -----------------------------------------
        # Step 2: Classify (LLM + ChromaDB)
        # -----------------------------------------
        t_cls_start = time.time()
        classification = self.classifier.classify(text, fname)
        t_cls_ms = (time.time() - t_cls_start) * 1000
        classification["original_filename"] = fname

        # -----------------------------------------
        # Step 3: File (Rename & Move)
        # -----------------------------------------
        t_file_start = time.time()
        if self.dry_run:
            new_filename = self.filer.build_new_filename(classification["tag"], fname)
            t_file_ms = (time.time() - t_file_start) * 1000
            t_total_ms = (time.time() - t_total_start) * 1000

            print(f"\n  [DRY RUN — no files moved]")
            print(f"  Tag        : {classification['tag']}")
            print(f"  Confidence : {classification['confidence']:.2f}")
            print(f"  New name   : {new_filename}")

            self._print_metrics(t_ext_ms, t_cls_ms, t_file_ms, t_total_ms)
            return {"success": True, "file": fname, "action": "dry_run"}

        # Actual filing
        result = self.filer.file_document(pdf_path, classification)
        t_file_ms = (time.time() - t_file_start) * 1000
        t_total_ms = (time.time() - t_total_start) * 1000

        # Summary Output
        print(f"\n  {result['message']}")
        self._print_metrics(t_ext_ms, t_cls_ms, t_file_ms, t_total_ms)

        return result

    def _print_metrics(self, ext_ms, cls_ms, file_ms, total_ms):
        """Helper to print formatted timing metrics."""
        print(f"\n  ⏱️ PERFORMANCE METRICS:")
        print(f"    Extraction : {ext_ms:8.2f} ms")
        print(f"    Classifier : {cls_ms:8.2f} ms")
        print(f"    Filing     : {file_ms:8.2f} ms")
        print(f"    -------------------------")
        print(f"    TOTAL TIME : {total_ms:8.2f} ms")
        print(f"{'─'*55}")

def process_folder(folder_path: str, debug: bool, dry_run: bool):
    folder = Path(folder_path)
    pdfs = sorted(folder.glob("*.pdf")) + sorted(folder.glob("*.PDF"))

    if not pdfs:
        print(f"[INFO] No PDFs found in: {folder_path}")
        return

    print(f"\n{'='*55}")
    print(f"  BATCH MODE — {len(pdfs)} PDF(s)")
    if dry_run: print(f"  Mode   : DRY RUN")
    print(f"{'='*55}")

    filer_inst = Filer(debug)
    auto_filer = DocumentAutoFiler(debug, dry_run)
    results = [auto_filer.process(str(pdf)) for pdf in pdfs]

    if not dry_run:
        filer_inst.save_log()

    print(f"\n{'='*55}")
    print(f"  BATCH COMPLETE")
    print(f"  Filed     : {sum(1 for r in results if r.get('action') == 'filed')}")
    print(f"  Uncertain : {sum(1 for r in results if r.get('action') == 'uncertain')}")
    print(f"  Errors    : {sum(1 for r in results if r.get('action') == 'error')}")
    print(f"{'='*55}")

def run():
    parser = argparse.ArgumentParser(description="GESDOC Document Classifier")
    parser.add_argument("--pdf", default=None, help="Single PDF to classify")
    parser.add_argument("--folder", default=None, help="Process all PDFs in folder")
    parser.add_argument("--dry-run", action="store_true", help="Classify but do NOT move")
    parser.add_argument("--debug", action="store_true", help="Detailed output")
    args = parser.parse_args()

    if args.folder:
        process_folder(args.folder, args.debug, args.dry_run)
        sys.exit(0)

    if args.pdf:
        if not os.path.exists(args.pdf):
            print(f"[ERROR] File not found: {args.pdf}")
            sys.exit(1)
        auto_filer = DocumentAutoFiler(args.debug, args.dry_run)
        result = auto_filer.process(args.pdf)
        if not args.dry_run: auto_filer.filer.save_log()
        sys.exit(0 if result.get("success") else 1)

    parser.print_help()

if __name__ == "__main__":
    run()