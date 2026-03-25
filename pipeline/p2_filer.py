"""
Project 2 — Document Auto-Filer
filer.py — rename file and move to correct folder

Single responsibility: take a classification result and physically
rename + move the file. Nothing else happens here.

Filename structure: TAG_IDENTIFIER_originalfilename.pdf
Example:           INVOICE_TECNOVA_sc_02_consultoria.pdf
"""
import os
import shutil
from pathlib import Path
from datetime import datetime
from config import Config


class Filer:
    """
    Handles the physical renaming and moving of classified PDFs.

    Safety features:
      - Never overwrites an existing file (adds timestamp suffix if conflict)
      - Creates target folder automatically if it doesn't exist
      - Validates new filename before attempting move
      - Logs every action taken for audit trail
    """

    def __init__(self, debug: bool = False):
        self.debug    = debug
        self._log     = []   # in-memory audit log for this session

    def build_new_filename(self, tag: str, identifier: str,
                           original_filename: str) -> str:
        """
        Build the new filename from classification result.

        Structure: TAG_IDENTIFIER_originalfilename.ext
        Example:   INVOICE_TECNOVA_sc_02_consultoria.pdf

        Rules:
          - All uppercase for TAG
          - Identifier cleaned (no spaces, no forbidden chars)
          - Original filename preserved as-is (just the stem, not the path)
          - Extension preserved
        """
        stem = Path(original_filename).stem
        ext  = Path(original_filename).suffix.lower()  # .pdf

        # Clean each component
        tag        = tag.upper().strip()
        identifier = identifier.replace(" ", "_").strip("_")

        # Remove any forbidden characters from the combined filename
        new_stem = f"{tag}_{identifier}_{stem}"
        for ch in Config.FILENAME_FORBIDDEN_CHARS:
            new_stem = new_stem.replace(ch, "")

        # Collapse multiple underscores
        while "__" in new_stem:
            new_stem = new_stem.replace("__", "_")

        return new_stem + ext

    def _resolve_conflict(self, target_path: Path) -> Path:
        """
        If target path already exists, add a timestamp suffix to avoid overwriting.
        Example: INVOICE_TECNOVA_file.pdf → INVOICE_TECNOVA_file_20260325_143022.pdf
        """
        if not target_path.exists():
            return target_path

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem      = target_path.stem
        suffix    = target_path.suffix
        return target_path.parent / f"{stem}_{timestamp}{suffix}"

    def file_document(self, source_path: str,
                      classification: dict) -> dict:
        """
        Rename and move a document based on its classification result.

        Args:
            source_path    : current full path to the PDF
            classification : result dict from Classifier.classify()

        Returns dict:
            success       : bool
            source        : original path
            destination   : new full path after move
            new_filename  : just the filename part
            action        : "filed", "uncertain", "error"
            message       : human-readable summary
        """
        source = Path(source_path)
        if not source.exists():
            return self._result(False, source_path, None,
                                "error", f"Source file not found: {source_path}")

        tag        = classification["tag"]
        identifier = classification["identifier"]
        folder     = classification["folder"]
        confidence = classification["confidence"]

        # Build new filename
        new_filename = self.build_new_filename(tag, identifier, source.name)

        # Build target folder path
        target_folder = Path(Config.OUTPUT_ROOT) / folder
        target_folder.mkdir(parents=True, exist_ok=True)

        # Build target path, resolve conflicts
        target_path = self._resolve_conflict(target_folder / new_filename)

        # Move the file
        try:
            shutil.move(str(source), str(target_path))
            action  = "uncertain" if classification["is_uncertain"] else "filed"
            message = (
                f"{'⚠️  UNCERTAIN — ' if classification['is_uncertain'] else '✅ Filed: '}"
                f"{new_filename} → {folder}/ "
                f"(confidence={confidence:.2f})"
            )
            if classification.get("is_new_type"):
                message += " [NEW DOCUMENT TYPE]"

            self._log_action(str(source), str(target_path), action, message)

            if self.debug:
                print(f"  [Filer] {message}")

            return self._result(True, str(source), str(target_path),
                                action, message, new_filename)

        except Exception as e:
            error_msg = f"Move failed: {e}"
            self._log_action(str(source), None, "error", error_msg)
            return self._result(False, str(source), None, "error", error_msg)

    def _log_action(self, source: str, destination: str | None,
                    action: str, message: str):
        """Append to in-memory audit log."""
        self._log.append({
            "timestamp":   datetime.now().isoformat(),
            "action":      action,
            "source":      source,
            "destination": destination,
            "message":     message,
        })

    def save_log(self, log_path: str = None):
        """
        Save audit log to a text file.
        Useful for tracking what was filed in each session.
        """
        import json
        if log_path is None:
            log_path = Path(Config.OUTPUT_ROOT) / "filing_log.jsonl"

        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            for entry in self._log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if self.debug:
            print(f"  [Filer] Log saved: {log_path} ({len(self._log)} entries)")

    @staticmethod
    def _result(success, source, destination, action, message,
                new_filename=None) -> dict:
        return {
            "success":      success,
            "source":       source,
            "destination":  destination,
            "new_filename": new_filename,
            "action":       action,
            "message":      message,
        }
