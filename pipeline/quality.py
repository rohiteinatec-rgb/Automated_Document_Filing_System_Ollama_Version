import re
from config import Config

class QualityGate:
    def evaluate(self, text: str, debug: bool = False) -> dict:
        issues = []
        fields_found = []
        score = 100

        if not text or not text.strip():
            return {"passed": False, "score": 0, "issues": ["empty output"], "fields_found": []}

        stripped = text.strip()

        # ── Layer 1: Generic Health ──
        if len(stripped) < 80:
            issues.append(f"suspiciously short: {len(stripped)} chars")
            score -= 40
        if "(cid:" in stripped:
            issues.append(f"font encoding corruption: {stripped.count('(cid:')} sequences")
            score -= 50

        # ── Layer 2: Glitch & Layout Integrity (The Golden Mean) ──
        # 1. Glitches: Digital files have 0. Scans have typos. We penalize EVERY glitch now.
        glitches = len(re.findall(r'\b\d+[a-zA-Z]+\d+\b', stripped))
        if glitches > 0:
            issues.append(f"OCR corruption: {glitches} alphanumeric glitches")
            score -= (15 * glitches)

        # 2. Table Layout: Since our Digital track explicitly builds Markdown tables,
        # a lack of pipes (|) means the layout collapsed (typical of raw EasyOCR).
        if stripped.count('|') < 4:
            issues.append("Table layout collapsed (no Markdown pipes)")
            score -= 20

        # ── Layer 3: Semantic Density ──
        for field_name, pattern in Config.INVOICE_FIELD_PATTERNS.items():
            if pattern.search(stripped):
                fields_found.append(field_name)

        if len(fields_found) < Config.MIN_FIELDS_REQUIRED:
            issues.append(f"too few fields: {len(fields_found)}/{len(Config.INVOICE_FIELD_PATTERNS)}")
            score -= 30

        if "iban" not in fields_found:
            issues.append("No IBAN found")
            score -= 20

            # Final Evaluation
        score = max(0, score)
        passed = score >= 80 and len(fields_found) >= Config.MIN_FIELDS_REQUIRED

        if debug:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"    [QualityGate] {status} — score={score} | fields={fields_found} | issues={issues}")

        return {"passed": passed, "score": score, "issues": issues, "fields_found": fields_found}