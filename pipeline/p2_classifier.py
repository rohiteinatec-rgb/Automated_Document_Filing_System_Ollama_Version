"""
Project 2 — Document Auto-Filer
classifier.py — LLM classification + ChromaDB tag memory

Two responsibilities:
  1. Classify the document and generate a consistent TAG
  2. Remember all tags ever used (ChromaDB) to prevent tag drift

Tag consistency problem solved:
  Without memory: "Nómina" → PAYSLIP today, NOMINA tomorrow, SALARY-SLIP next week
  With ChromaDB:  LLM is shown existing tags first → reuses PAYSLIP every time
"""
import json
import re
import requests

try:
    import chromadb
    CHROMA_OK = True
except ImportError:
    CHROMA_OK = False
    print("[WARN] chromadb not installed: pip install chromadb")

from config import Config


# ── Result dataclass (plain dict for simplicity) ──────────────
# Every classification returns this structure:
#   tag          : str  — e.g. "INVOICE", "MODEL-111", "PAYSLIP"
#   identifier   : str  — e.g. "TECNOVA", "Rohit_Kumar", "Q1-2026"
#   confidence   : float 0.0-1.0
#   is_new_type  : bool — True if tag was not in ChromaDB before
#   reasoning    : str  — LLM's explanation (for debug/audit)


class TagMemory:
    """
    ChromaDB-backed memory of all tags ever used.
    Ensures the LLM reuses existing tags instead of inventing variants.
    """

    def __init__(self, debug: bool = False):
        self._collection = None
        self._debug      = debug
        if CHROMA_OK:
            try:
                client           = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
                self._collection = client.get_or_create_collection(Config.CHROMA_COLLECTION)
                if debug:
                    print(f"  [TagMemory] Ready — {self._collection.count()} tags stored")
            except Exception as e:
                print(f"  [TagMemory] ChromaDB init failed: {e}")

    def get_existing_tags(self) -> list:
        """Return all tags previously used, with their descriptions."""
        if not self._collection or self._collection.count() == 0:
            return []
        try:
            results = self._collection.get(include=["metadatas"])
            tags = []
            for meta in results["metadatas"]:
                tags.append({
                    "tag":         meta.get("tag", ""),
                    "description": meta.get("description", ""),
                    "example":     meta.get("example_text", ""),
                    "count":       meta.get("count", 1),
                })
            return tags
        except Exception as e:
            if self._debug:
                print(f"  [TagMemory] Failed to retrieve tags: {e}")
            return []

    def store_tag(self, tag: str, description: str, example_text: str):
        """
        Store a new tag or increment count if it already exists.
        Uses the tag string as the document ID so duplicates are handled
        automatically via upsert.
        """
        if not self._collection:
            return
        try:
            # Check if tag already exists to increment count
            existing = self._collection.get(ids=[tag])
            if existing["ids"]:
                current_count = existing["metadatas"][0].get("count", 1)
                self._collection.upsert(
                    ids=[tag],
                    documents=[description],
                    metadatas=[{
                        "tag":          tag,
                        "description":  description,
                        "example_text": example_text[:300],
                        "count":        current_count + 1,
                    }]
                )
                if self._debug:
                    print(f"  [TagMemory] Updated tag '{tag}' "
                          f"(seen {current_count + 1} times)")
            else:
                self._collection.add(
                    ids=[tag],
                    documents=[description],
                    metadatas=[{
                        "tag":          tag,
                        "description":  description,
                        "example_text": example_text[:300],
                        "count":        1,
                    }]
                )
                if self._debug:
                    print(f"  [TagMemory] New tag stored: '{tag}'")
        except Exception as e:
            if self._debug:
                print(f"  [TagMemory] Store failed: {e}")


class Classifier:
    """
    LLM-based document classifier with tag memory.

    Flow:
      1. Load existing tags from ChromaDB
      2. Build prompt with known types + existing tags + document text
      3. LLM returns JSON: {tag, identifier, confidence, reasoning}
      4. Validate and clean response
      5. Store new tag in ChromaDB if genuinely new
    """

    def __init__(self, debug: bool = False):
        self.debug      = debug
        self.tag_memory = TagMemory(debug)

    def _check_ollama(self) -> bool:
        try:
            r      = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags", timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            return any(Config.OLLAMA_MODEL.split(":")[0] in m for m in models)
        except Exception:
            return False

    def _build_prompt(self, text: str, existing_tags: list) -> str:
        """
        Build the classification prompt.

        Three sections:
          A. Known document types (always present — INVOICE, WORK-CONTRACT)
          B. Tags already used (from ChromaDB — for consistency)
          C. The document text to classify
        """
        # Section A: known types
        known_section = ""
        for tag, info in Config.KNOWN_TYPES.items():
            sp = ", ".join(f'"{t}"' for t in info["spanish_terms"][:3])
            ca = ", ".join(f'"{t}"' for t in info["catalan_terms"][:2])
            known_section += (
                f'  - TAG="{tag}": {info["description"]}\n'
                f'    Spanish signals: {sp}\n'
                f'    Catalan signals: {ca}\n'
            )

        # Section B: previously used tags
        memory_section = ""
        if existing_tags:
            memory_section = "\nPREVIOUSLY USED TAGS (reuse these if the document matches):\n"
            for t in existing_tags:
                count_str = f" (seen {t['count']} times)" if t.get("count", 1) > 1 else ""
                memory_section += f'  - TAG="{t["tag"]}": {t["description"]}{count_str}\n'

        return f"""You are a document classification expert for a Spanish/Catalan company.
Analyse the document text and return a JSON classification.

KNOWN DOCUMENT TYPES (always check these first):
{known_section}{memory_section}
CLASSIFICATION RULES:
1. Check KNOWN TYPES first. If the document matches, use that exact TAG.
2. Check PREVIOUSLY USED TAGS second. Reuse an existing tag if the document type matches.
3. Only invent a NEW tag if the document genuinely does not match any existing type.
4. TAG format: UPPERCASE, hyphens for spaces, no special characters. Max 30 chars.
   Examples: INVOICE, WORK-CONTRACT, MODEL-111, PAYSLIP, DELIVERY-NOTE, BANK-STATEMENT
5. IDENTIFIER: extract the single most useful identifier for filing:
   - For INVOICE: supplier/vendor name (e.g. "TECNOVA", "CaixaBank")
   - For WORK-CONTRACT: employee full name (e.g. "Rohit_Kumar")
   - For tax models: period (e.g. "Q1-2026", "2025-T3")
   - For others: whatever best identifies this specific document
   - Use underscores instead of spaces. Max 40 chars.
6. CONFIDENCE: 0.0-1.0. Be honest.
   - 0.9+: document clearly matches a type with strong signals
   - 0.75-0.9: good match but some ambiguity
   - Below 0.75: uncertain — document will go to UNCERTAIN folder
7. LANGUAGE: documents may be in Spanish or Catalan — classify correctly either way.
   "Factura" (Spanish) = "Factura" (Catalan) = INVOICE
   "Contrato de trabajo" (Spanish) = "Contracte de treball" (Catalan) = WORK-CONTRACT

DOCUMENT TEXT:
---
{text[:1400]}
---

Respond ONLY with valid JSON, no markdown, no explanation:
{{
  "tag": "TAG_HERE",
  "identifier": "identifier_here",
  "confidence": 0.0,
  "reasoning": "brief explanation of why this tag was chosen"
}}"""

    def _parse_response(self, raw: str) -> dict | None:
        """Parse and validate LLM JSON response."""
        # Strip markdown fences if present
        cleaned = raw.strip()
        if "```" in cleaned:
            cleaned = "\n".join(
                l for l in cleaned.splitlines()
                if not l.strip().startswith("```")
            ).strip()

        s = cleaned.find("{")
        e = cleaned.rfind("}") + 1
        if s == -1 or e == 0:
            return None

        try:
            data = json.loads(cleaned[s:e])
        except json.JSONDecodeError:
            return None

        # Validate required fields
        if not all(k in data for k in ["tag", "identifier", "confidence"]):
            return None

        # Sanitise tag — uppercase, hyphens only, max length
        tag = str(data["tag"]).upper().strip()
        tag = re.sub(r'[^A-Z0-9\-]', '-', tag)
        tag = re.sub(r'-+', '-', tag).strip('-')
        tag = tag[:Config.MAX_TAG_LENGTH]

        # Sanitise identifier — no forbidden chars, underscores for spaces
        identifier = str(data.get("identifier", "doc")).strip()
        for ch in Config.FILENAME_FORBIDDEN_CHARS:
            identifier = identifier.replace(ch, "")
        identifier = identifier.replace(" ", "_")
        identifier = identifier[:Config.MAX_IDENTIFIER_LENGTH]
        if not identifier:
            identifier = "doc"

        # Validate confidence
        try:
            confidence = float(data["confidence"])
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.5

        return {
            "tag":        tag,
            "identifier": identifier,
            "confidence": confidence,
            "reasoning":  str(data.get("reasoning", ""))[:200],
        }

    def classify(self, text: str, original_filename: str) -> dict:
        """
        Classify a document and return filing instructions.

        Returns dict:
            tag          : str   — e.g. "INVOICE"
            identifier   : str   — e.g. "TECNOVA"
            confidence   : float — 0.0-1.0
            is_uncertain : bool  — True if below confidence threshold
            is_new_type  : bool  — True if tag not seen before
            reasoning    : str
            folder       : str   — target folder name
        """
        if self.debug:
            print(f"  [Classifier] Classifying '{original_filename}'...")

        # Handle empty text
        if not text or len(text.strip()) < 20:
            return self._uncertain_result("insufficient text extracted")

        # Check Ollama
        if not self._check_ollama():
            return self._uncertain_result("Ollama not available")

        # Load tag memory
        existing_tags = self.tag_memory.get_existing_tags()
        if self.debug:
            print(f"  [Classifier] {len(existing_tags)} existing tags in memory")

        # Build and send prompt
        prompt = self._build_prompt(text, existing_tags)
        try:
            resp = requests.post(
                f"{Config.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model":  Config.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature":  0.01,   # near-deterministic
                        "top_p":        0.9,
                        "num_predict":  Config.OLLAMA_NUM_PREDICT,
                    }
                },
                timeout=Config.OLLAMA_TIMEOUT
            )
            resp.raise_for_status()
            raw_response = resp.json().get("response", "")

            if self.debug:
                print(f"  [Classifier] Raw response: {raw_response[:200]}")

        except requests.exceptions.Timeout:
            return self._uncertain_result("Ollama timeout")
        except Exception as e:
            return self._uncertain_result(f"Ollama error: {e}")

        # Parse response
        parsed = self._parse_response(raw_response)
        if not parsed:
            return self._uncertain_result("could not parse LLM response")

        tag        = parsed["tag"]
        confidence = parsed["confidence"]

        # Determine if this is a new type (not in known or memory)
        known_tags    = set(Config.KNOWN_TYPES.keys())
        existing_tags_set = {t["tag"] for t in existing_tags}
        is_new_type   = tag not in known_tags and tag not in existing_tags_set

        # Determine if uncertain
        is_uncertain = confidence < Config.CONFIDENCE_THRESHOLD

        # Determine target folder
        if is_uncertain:
            folder = Config.UNCERTAIN_FOLDER
        elif tag in Config.KNOWN_TYPES:
            folder = Config.KNOWN_TYPES[tag]["folder"]
        else:
            folder = tag  # new type gets its own folder named after the tag

        # Store in tag memory (even uncertain ones — helps future classification)
        description = parsed["reasoning"] or f"Document classified as {tag}"
        self.tag_memory.store_tag(tag, description, text[:300])

        result = {
            "tag":         tag,
            "identifier":  parsed["identifier"],
            "confidence":  confidence,
            "is_uncertain":is_uncertain,
            "is_new_type": is_new_type,
            "reasoning":   parsed["reasoning"],
            "folder":      folder,
        }

        if self.debug:
            status = "⚠️  UNCERTAIN" if is_uncertain else "✅"
            new    = " (NEW TYPE)" if is_new_type else ""
            print(f"  [Classifier] {status} TAG={tag}{new} | "
                  f"id={parsed['identifier']} | "
                  f"confidence={confidence:.2f} | "
                  f"→ {folder}/")

        return result

    @staticmethod
    def _uncertain_result(reason: str) -> dict:
        return {
            "tag":         "UNCERTAIN",
            "identifier":  "doc",
            "confidence":  0.0,
            "is_uncertain":True,
            "is_new_type": False,
            "reasoning":   reason,
            "folder":      Config.UNCERTAIN_FOLDER,
        }
