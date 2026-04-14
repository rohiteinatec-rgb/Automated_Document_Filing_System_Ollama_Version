import os

class Config:
    # -- Ollama CPU Optimization --
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "qwen3:8b"
    OLLAMA_TIMEOUT = 180  # 3 minutes for CPU inference

    # Critical CPU Performance Tweaks
    OLLAMA_OPTIONS = {
        "num_ctx": 2048,      # Small context = faster CPU math
        "temperature": 0.0,   # Deterministic
        "num_thread": 12,     # Adjust to your physical core count
        "num_predict": 150    # Short responses save CPU cycles
    }

    # -- Paths --
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHROMA_DB_PATH = os.path.join(BASE_DIR, "chromadb")
    OUTPUT_ROOT = os.path.join(BASE_DIR, "output")

    # -- Digital-Only Constraints --
    CHARS_FOR_CLASSIFICATION = 1200 # Limit text sent to CPU LLM
    CONFIDENCE_THRESHOLD = 0.70
    MAX_TAG_LENGTH = 30

    # -- Filer Constraints (Added to fix filer.py errors) --
    KNOWN_TAG_PREFIXES = ["invoice", "factura", "nomina", "work-contract", "m111", "other", "uncertain"]
    FILENAME_FORBIDDEN_CHARS = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']

    # Folder Mapping
    TAG_FOLDERS = {
        "factura": "school-financial",
        "invoice": "school-financial",
        "nomina": "hr-payroll",
        "work-contract": "hr-contracts",
        "m111": "tax-M111",
        "other": "unclassified",
        "uncertain": "UNCERTAIN"
    }

    @classmethod
    def get_folder(cls, tag: str) -> str:
        return cls.TAG_FOLDERS.get(tag.lower(), "unclassified")

    @classmethod
    def get_all_tags(cls) -> list:
        return list(cls.TAG_FOLDERS.keys())