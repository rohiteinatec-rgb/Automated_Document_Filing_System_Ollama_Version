"""
Project 2 — Document Auto-Filer
config.py — all configuration in one place
"""
import os

class Config:

    # ── Ollama ────────────────────────────────────────────────
    OLLAMA_BASE_URL  = "http://localhost:11434"
    # Text-only classification — no vision needed for 95% digital docs.
    # Use a fast text model if available, otherwise llama3.2-vision works too.
    OLLAMA_MODEL     = "Mistral Small 3"
    OLLAMA_TIMEOUT   = 120   # classification is fast — 120s is generous
    OLLAMA_NUM_PREDICT = 200 # tag + identifier only — never needs more than 200 tokens

    # ── Paths ─────────────────────────────────────────────────
    # Where PDFs land for processing
    INPUT_FOLDER  = r"C:\Project2\input"

    # Root of the filing structure — subfolders created automatically per tag
    OUTPUT_ROOT   = r"C:\Project2\filed"

    # ChromaDB for tag memory — remembers every tag ever used
    CHROMA_DB_PATH   = r"C:\Project2\chromadb"
    CHROMA_COLLECTION = "document_tags"

    # ── Classification ────────────────────────────────────────
    # Confidence threshold 0.0-1.0.
    # Below this → file goes to UNCERTAIN folder for human review.
    # 0.75 is a good starting point — adjust after testing.
    CONFIDENCE_THRESHOLD = 0.75

    # Folder name for documents the LLM is not confident about
    UNCERTAIN_FOLDER = "UNCERTAIN"

    # How many characters to extract from the PDF for classification.
    # First 1500 chars usually contains the header with document type,
    # supplier name, date, and key identifiers. More than enough.
    CHARS_FOR_CLASSIFICATION = 1500

    # ── Known document types (seed knowledge for the LLM) ─────
    # These are always known. LLM will invent tags for everything else.
    # Keys = tag the LLM should output. Values = description + Spanish terms
    # so the LLM can match Spanish documents correctly.
    KNOWN_TYPES = {
        "INVOICE": {
            "description": "Commercial invoice / bill for goods or services",
            "spanish_terms": ["FACTURA", "factura", "FACT/", "F/"],
            "catalan_terms": ["FACTURA", "factura"],
            "folder": "INVOICE",
        },
        "WORK-CONTRACT": {
            "description": "Employment or service contract between a company and a person",
            "spanish_terms": ["CONTRATO DE TRABAJO", "CONTRATO LABORAL",
                              "contrato de trabajo", "CONTRATO DE PRESTACIÓN"],
            "catalan_terms": ["CONTRACTE DE TREBALL", "contracte de treball"],
            "folder": "WORK-CONTRACT",
        },
    }

    # ── File safety ───────────────────────────────────────────
    # Characters not allowed in filenames on Windows/Linux
    FILENAME_FORBIDDEN_CHARS = r'\/:*?"<>|'

    # Maximum length for the generated tag (prevents runaway LLM output)
    MAX_TAG_LENGTH = 30

    # Maximum length for the identifier extracted by LLM (supplier name etc)
    MAX_IDENTIFIER_LENGTH = 40
