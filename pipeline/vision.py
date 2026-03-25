import requests
from config import Config

class VisionAI:
    @staticmethod
    def _remove_loops(text: str, max_chars: int = 1800) -> str:
        """Removes hallucination repetition loops."""
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

    def correct_document(self, images_b64: list, easyocr_text: str, debug: bool = False) -> str:
        corrected_pages = []
        for i, img_b64 in enumerate(images_b64):
            context = easyocr_text[:1200]

            prompt = f"""You are a precise data extraction AI analyzing a Spanish invoice. Compare the attached image with this flawed OCR text:
            
            --- OCR TEXT ---
            {context}
            --- END OCR TEXT ---
            
            Produce a flawlessly CORRECTED and COMPLETE Markdown version of the invoice. 
            CRITICAL INSTRUCTIONS:
            1. FIX CORRUPTED NUMBERS: Correct letters mixed into numbers (e.g., "6.7oo.00" -> "6.700,00").
            2. FIX PERCENTAGES: Correct misread symbols (e.g., "IVA (2186)" -> "IVA (21%)").
            3. REBUILD THE TABLE: Ensure no financial values are left blank.
            4. EXTRACT THE FOOTER: MUST extract bank details, IBAN (starts with ES), and addresses.
            5. Preserve Spanish terminology exactly. Output ONLY Markdown."""

            try:
                if debug: print(f"    [VisionAI] Correcting page {i+1}...")
                resp = requests.post(
                    f"{Config.OLLAMA_BASE_URL}/api/generate",
                    json={"model": Config.OLLAMA_MODEL, "prompt": prompt, "images": [img_b64], "stream": False, "options": {"temperature": 0.01}},
                    timeout=Config.OLLAMA_TIMEOUT
                )
                resp.raise_for_status()
                corrected = self._remove_loops(resp.json().get("response", "").strip())
                corrected_pages.append(corrected)
            except Exception as e:
                if debug: print(f"    [VisionAI] Error on page {i+1}: {e}")
                corrected_pages.append(easyocr_text)

        return "\n\n".join(corrected_pages)