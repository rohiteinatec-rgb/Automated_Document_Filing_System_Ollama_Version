"""
GESDOC Document Classifier
filer.py — rename file and move to correct folder

Single responsibility: take a classification result and physically
rename + move the file. Nothing else happens here.

Filename structure: TAG_originalfilename.pdf
Example:           invoice_document.pdf
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
        self.debug = debug
        self._log = []

    def strip_existing_tag(self, filename: str) -> str:
        """
        Strip existing tag prefix from filename to prevent double-tagging.
        Example: invoice_test.pdf → test.pdf
                work-contract_doc.pdf → doc.pdf
        """
        stem = Path(filename).stem
        
        for prefix in Config.KNOWN_TAG_PREFIXES:
            # Check if filename starts with prefix_
            if stem.lower().startswith(prefix + "_"):
                stem = stem[len(prefix) + 1:]
                break
            # Check if contains _ prefix pattern
            parts = stem.split("_")
            if len(parts) > 1 and parts[0].lower() in Config.KNOWN_TAG_PREFIXES:
                stem = "_".join(parts[1:])
                break
        
        return stem

    def build_new_filename(self, tag: str, original_filename: str) -> str:
        """
        Build the new filename from classification result.

        Structure: TAG_originalfilename.ext
        Example:   invoice_test_document.pdf

        Rules:
          - Tag is lowercase (as per new spec)
          - Original filename stripped of any existing tag prefix
          - Extension preserved
        """
        # Strip existing tag prefix
        clean_stem = self.strip_existing_tag(original_filename)
        ext = Path(original_filename).suffix.lower()
        
        # Clean tag
        tag = tag.lower().strip()
        
        # Remove forbidden characters
        new_stem = f"{tag}_{clean_stem}"
        for ch in Config.FILENAME_FORBIDDEN_CHARS:
            new_stem = new_stem.replace(ch, "")
        
        # Collapse multiple underscores
        while "__" in new_stem:
            new_stem = new_stem.replace("__", "_")
        
        return new_stem + ext

    def _resolve_conflict(self, target_path: Path) -> Path:
        """
        If target path already exists, add a timestamp suffix to avoid overwriting.
        Example: invoice_test.pdf → invoice_test_20260327_143022.pdf
        """
        if not target_path.exists():
            return target_path

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = target_path.stem
        suffix = target_path.suffix
        return target_path.parent / f"{stem}_{timestamp}{suffix}"

    def file_document(self, source_path: str,
                      classification: dict) -> dict:
        """
        Rename and move a document based on its classification result.
        """
        source = Path(source_path)
        if not source.exists():
            return self._result(False, source_path, None,
                               "error", f"Source file not found: {source_path}")

        tag = classification["tag"]
        folder = classification["folder"]
        confidence = classification.get("confidence", 0.0)
        original_filename = classification.get("original_filename", source.name)

        # Build new filename (no identifier in new spec)
        new_filename = self.build_new_filename(tag, original_filename)

        # Build target folder path (use folder from classification)
        target_folder = Path(Config.OUTPUT_ROOT) / folder
        target_folder.mkdir(parents=True, exist_ok=True)

        # Build target path, resolve conflicts
        target_path = self._resolve_conflict(target_folder / new_filename)

        # Move the file
        try:
            shutil.move(str(source), str(target_path))
            action = "uncertain" if classification.get("is_uncertain") else "filed"
            message = (
                f"{'⚠️  UNCERTAIN — ' if classification.get('is_uncertain') else '✅ Filed: '}"
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
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "source": source,
            "destination": destination,
            "message": message,
        })

    def save_log(self, log_path: str = None):
        """
        Save audit log to a text file.
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
            "success": success,
            "source": source,
            "destination": destination,
            "new_filename": new_filename,
            "action": action,
            "message": message,
        }
