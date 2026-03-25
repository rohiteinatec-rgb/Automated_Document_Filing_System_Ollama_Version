import re

class Config:
    OLLAMA_BASE_URL  = "http://localhost:11434"
    OLLAMA_MODEL     = "llama3.2-vision:latest"
    OLLAMA_TIMEOUT   = 300
    RENDER_DPI       = 300
    OUTPUT_FOLDER    = "output"
    DEFAULT_LANGS    = ["es", "en"]
    MIN_FIELDS_REQUIRED = 2

    INVOICE_FIELD_PATTERNS = {
        "cif_nie":    re.compile(r'\b[A-Z]\d{7}[A-Z0-9]\b'),
        "iban":       re.compile(r'\bES\d{2}[\s\d]{20,26}\b', re.IGNORECASE),
        "euro_amount":re.compile(r'\d{1,3}(?:[.,]\d{3})*[.,]\d{2}\s*€'),
        "date":       re.compile(r'\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b'),
        "invoice_num":re.compile(r'\b[A-Z]{2,6}[/\-]\d{4}[/\-]\d{3,6}\b'),
    }