# Save this as quality.py in your pipeline folder
import re

class QualityGate:
    def __init__(self, debug: bool = False):
        self.debug = debug

    def evaluate(self, text: str, debug: bool = False) -> dict:
        """Simple quality check to see if extraction yielded usable text."""
        issues = []
        score = 100

        if not text or len(text.strip()) < 50:
            return {"passed": False, "score": 0, "issues": ["Empty or too short"]}

        # Check for font corruption
        if "(cid:" in text:
            issues.append("Font corruption detected")
            score -= 50

        passed = score >= 50
        return {"passed": passed, "score": score, "issues": issues}