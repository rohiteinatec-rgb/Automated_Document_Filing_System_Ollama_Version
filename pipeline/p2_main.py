"""
Project 2 — Document Auto-Filer
main.py — orchestrator and CLI entry point

USAGE:
  # Single PDF
  python main.py --pdf C:/Project2/input/invoice.pdf

  # Single PDF with debug output
  python main.py --pdf C:/Project2/input/invoice.pdf --debug

  # Watch input folder — process all PDFs found there
  python main.py --folder C:/Project2/input

  # Show all tags stored in memory
  python main.py --tags

  # Dry run — classify but do NOT move files
  python main.py --pdf invoice.pdf --dry-run

FOLDER STRUCTURE CREATED AUTOMATICALLY:
  C:/Project2/filed/
    INVOICE/
    WORK-CONTRACT/
    MODEL-111/         ← created when first MODEL-111 doc is seen
    PAYSLIP/           ← created when first payslip is seen
    UNCERTAIN/         ← low-confidence documents for human review
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
    """
    Orchestrates the full pipeline for one PDF:
      1. Read  — extract text with pymuPDF (fast, no AI)
      2. Classify — LLM identifies document type + key identifier
      3. File  — rename and move to correct folder
    """

    def __init__(self, debug: bool = False, dry_run: bool = False):
        self.debug      = debug
        self.dry_run    = dry_run
        self.classifier = Classifier(debug)
        self.filer      = Filer(debug)

    def process(self, pdf_path: str) -> dict:
        """
        Process a single PDF through the full pipeline.

        Returns result dict with all details of what happened.
        """
        t_start  = time.time()
        pdf_path = str(pdf_path)
        fname    = Path(pdf_path).name

        print(f"\n{'─'*55}")
        print(f"  📄 {fname}")
        print(f"{'─'*55}")

        # ── Step 1: Read ──────────────────────────────────────
        if not PDFReader.is_readable(pdf_path):
            print(f"  ❌ Cannot read file — skipping")
            return {"success": False, "file": fname, "action": "error",
                    "message": "unreadable PDF"}

        text, method = PDFReader.extract_for_classification(pdf_path, self.debug)

        if self.debug:
            print(f"  [Read] Method={method} | {len(text)} chars extracted")
            if text:
                preview = text[:150].replace('\n', ' ')
                print(f"  [Read] Preview: {preview}...")

        # ── Step 2: Classify ─────────────────────────────────
        classification = self.classifier.classify(text, fname)

        # ── Step 3: File (or dry run) ─────────────────────────
        if self.dry_run:
            new_filename = self.filer.build_new_filename(
                classification["tag"],
                classification["identifier"],
                fname
            )
            folder  = classification["folder"]
            elapsed = time.time() - t_start
            print(f"\n  [DRY RUN — no files moved]")
            print(f"  Tag         : {classification['tag']}")
            print(f"  Identifier  : {classification['identifier']}")
            print(f"  Confidence  : {classification['confidence']:.2f}")
            print(f"  New name    : {new_filename}")
            print(f"  Target      : {Config.OUTPUT_ROOT}/{folder}/")
            print(f"  Reasoning   : {classification['reasoning']}")
            if classification["is_uncertain"]:
                print(f"  ⚠️  Below confidence threshold → UNCERTAIN folder")
            if classification["is_new_type"]:
                print(f"  🆕 NEW document type discovered")
            print(f"  Time        : {elapsed:.1f}s")
            return {"success": True, "file": fname, "action": "dry_run",
                    "classification": classification}

        result  = self.filer.file_document(pdf_path, classification)
        elapsed = time.time() - t_start

        # ── Summary ───────────────────────────────────────────
        print(f"\n  {result['message']}")
        if classification["is_new_type"]:
            print(f"  🆕 NEW document type '{classification['tag']}' "
                  f"— folder created automatically")
        print(f"  Time : {elapsed:.1f}s")

        return result


def process_folder(folder_path: str, debug: bool, dry_run: bool):
    """Process all PDFs found in a folder."""
    folder = Path(folder_path)
    pdfs   = sorted(folder.glob("*.pdf")) + sorted(folder.glob("*.PDF"))

    if not pdfs:
        print(f"[INFO] No PDFs found in: {folder_path}")
        return

    print(f"\n{'='*55}")
    print(f"  BATCH MODE — {len(pdfs)} PDF(s)")
    print(f"  Input  : {folder_path}")
    print(f"  Output : {Config.OUTPUT_ROOT}")
    if dry_run:
        print(f"  Mode   : DRY RUN (no files will be moved)")
    print(f"{'='*55}")

    filer_inst   = Filer(debug)
    auto_filer   = DocumentAutoFiler(debug, dry_run)
    results      = []

    for pdf in pdfs:
        result = auto_filer.process(str(pdf))
        results.append(result)

    # Save audit log
    if not dry_run:
        filer_inst.save_log()

    # Summary
    filed     = sum(1 for r in results if r.get("action") == "filed")
    uncertain = sum(1 for r in results if r.get("action") == "uncertain")
    errors    = sum(1 for r in results if r.get("action") == "error")

    print(f"\n{'='*55}")
    print(f"  BATCH COMPLETE")
    print(f"  Filed         : {filed}")
    print(f"  Uncertain     : {uncertain} → {Config.UNCERTAIN_FOLDER}/")
    print(f"  Errors        : {errors}")
    print(f"{'='*55}")


def show_tags():
    """Display all document types stored in tag memory."""
    from classifier import TagMemory
    memory = TagMemory(debug=False)
    tags   = memory.get_existing_tags()

    if not tags:
        print("[TagMemory] Empty — no documents classified yet.")
        return

    print(f"\n{'='*55}")
    print(f"  TAG MEMORY — {len(tags)} document types known")
    print(f"{'='*55}")
    for t in sorted(tags, key=lambda x: -x.get("count", 0)):
        print(f"  {t['tag']:25s} seen {t.get('count',1):3d}x")
        print(f"    → {t['description'][:60]}")
    print()


def run():
    parser = argparse.ArgumentParser(
        description="Project 2 — Document Auto-Filer"
    )
    parser.add_argument("--pdf",     default=None,
                        help="Single PDF to classify and file")
    parser.add_argument("--folder",  default=None,
                        help="Process all PDFs in this folder")
    parser.add_argument("--tags",    action="store_true",
                        help="Show all document types in tag memory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Classify but do NOT move files")
    parser.add_argument("--debug",   action="store_true",
                        help="Detailed step-by-step output")
    args = parser.parse_args()

    if args.tags:
        show_tags()
        sys.exit(0)

    if args.folder:
        process_folder(args.folder, args.debug, args.dry_run)
        sys.exit(0)

    if args.pdf:
        if not os.path.exists(args.pdf):
            print(f"[ERROR] File not found: {args.pdf}")
            sys.exit(1)
        auto_filer = DocumentAutoFiler(args.debug, args.dry_run)
        result     = auto_filer.process(args.pdf)
        if not args.dry_run:
            auto_filer.filer.save_log()
        sys.exit(0 if result.get("success") else 1)

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    run()
