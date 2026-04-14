import json
import requests
import hashlib
from config import Config

class TagMemory:
    """ChromaDB-backed memory to prevent tag drift on CPU."""
    def __init__(self, debug: bool = False):
        self._collection = None
        self._debug = debug
        try:
            import chromadb
            client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
            self._collection = client.get_or_create_collection(
                name="tag_metadata",
                metadata={"hnsw:space": "cosine"}
            )
            # Add this print to confirm connection!
            if self._debug:
                print(f"  [TagMemory] Connected. Total stored decisions: {self._collection.count()}")
        except Exception as e:
            if self._debug: print(f"  [TagMemory] Init failed: {e}")

    def get_existing_tags(self, text: str):
        if not self._collection or self._collection.count() == 0:
            if self._debug: print("  [TagMemory] Memory is empty. Skipping search.")
            return []
        try:
            results = self._collection.query(
                query_texts=[text[:500]],
                n_results=1,
                include=["metadatas", "distances"]
            )
            if results["metadatas"][0]:
                meta = results["metadatas"][0][0]
                dist = results["distances"][0][0]
                similarity = 1 - dist

                # Add this print to show what it found!
                if self._debug:
                    print(f"  [TagMemory] Found similar past decision: '{meta['tag']}' (Similarity: {similarity:.2f})")

                return [{"tag": meta["tag"], "similarity": similarity}]
        except Exception:
            pass
        return []

    def store_tag(self, tag: str, description: str, text: str):
        if not self._collection: return
        doc_id = hashlib.md5(text[:300].encode()).hexdigest()
        self._collection.upsert(
            ids=[doc_id],
            documents=[text[:500]],
            metadatas=[{"tag": tag, "desc": description}]
        )
        # Add this print to confirm it learned the new document!
        if self._debug:
            print(f"  [TagMemory] Learned new document. Stored tag: '{tag}'")

class Classifier:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.tag_memory = TagMemory(debug)

    def _call_ollama(self, prompt: str):
        """CPU-Optimized call to Qwen3:14B."""
        try:
            resp = requests.post(
                f"{Config.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": Config.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "keep_alive": -1, # Keep 14B model in your 128GB RAM
                    "options": Config.OLLAMA_OPTIONS
                },
                timeout=Config.OLLAMA_TIMEOUT
            )
            resp.raise_for_status()
            return resp.json().get("response", "{}")
        except Exception as e:
            if self.debug: print(f"  [Classifier] LLM Error: {e}")
            return None

    def classify(self, text: str, original_filename: str) -> dict:
        if self.debug: print(f"  [Classifier] CPU Classifying '{original_filename}'...")

        # 1. Check Memory Similarity first (Fast on CPU)
        existing_tags = self.tag_memory.get_existing_tags(text)

        # 2. Build Prompt & Call LLM
        prompt = self._build_prompt(text, existing_tags)
        raw_response = self._call_ollama(prompt)
        parsed = self._parse_response(raw_response) if raw_response else None

        # 3. SMART FALLBACK: If CPU timed out or returned empty {}
        if not parsed or not parsed.get("tag"):
            best_sim = 0.0
            best_tag = "other"
            if existing_tags:
                best_tag = existing_tags[0].get("tag", "other")
                best_sim = existing_tags[0].get("similarity", 0.0)

            if best_sim > 0.65:
                if self.debug: print(f"  [Classifier] LLM Timeout. Fallback to Memory: {best_tag}")
                parsed = {"tag": best_tag, "confidence": best_sim, "reasoning": "Memory fallback"}
            else:
                return self._uncertain_result("LLM timeout and no memory match")

        tag = str(parsed.get("tag", "other")).lower()

        # Safe float conversion for confidence
        raw_conf = parsed.get("confidence")
        confidence = float(raw_conf) if raw_conf is not None else 0.0

        # Final Folder Assignment
        folder = Config.get_folder(tag)

        if confidence > 0.5:
            self.tag_memory.store_tag(tag, "Decision stored", text[:300])

        return {
            "tag": tag,
            "confidence": confidence,
            "folder": folder,
            "is_uncertain": confidence < Config.CONFIDENCE_THRESHOLD,
            "original_filename": original_filename
        }

    def _build_prompt(self, text: str, similar: list) -> str:
        sim_text = "\n".join([f"- {s['tag']} (Similarity: {s['similarity']:.2f})" for s in similar]) if similar else "None"
        return f"""You are a document classifier. Read the text and output a JSON object.
        Allowed Tags: {', '.join(Config.KNOWN_TAG_PREFIXES)}
        
        Rules:
        1. Always return a valid JSON object: {{"tag": "name", "confidence": 0.0}}
        2. Do NOT return an empty object {{}}.
        
        Context Text:
        ---
        {text[:1000]}
        ---
        
        Similar past tags in database:
        {sim_text}
        """

    def _parse_response(self, raw: str) -> dict | None:
        try:
            # Strip markdown code blocks if the model wrapped the JSON
            cleaned = raw.strip()
            if "```" in cleaned:
                cleaned = "\n".join(l for l in cleaned.splitlines() if not l.strip().startswith("```")).strip()

            s = cleaned.find("{")
            e = cleaned.rfind("}") + 1
            if s == -1 or e == 0: return None

            return json.loads(cleaned[s:e])
        except json.JSONDecodeError:
            return None

    def _uncertain_result(self, reason: str) -> dict:
        return {
            "tag": "uncertain",
            "confidence": 0.0,
            "folder": "UNCERTAIN",
            "is_uncertain": True,
            "reasoning": reason,
            "original_filename": "doc.pdf"
        }