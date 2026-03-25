Project 2: Intelligent Document Router (IDR)
An automated, AI-driven document classification and filing system designed for European business environments. It utilizes Mistral Small 3 for high-precision text classification and Llama 3.2 Vision as a fallback for scanned documents.

🚀 Overview
Project 2 transforms a chaotic input directory into a structured, searchable archive. It identifies document types (Invoices, Contracts, Tax Models), extracts key identifiers (Provider names, Dates), renames files, and routes them to the appropriate directories.

Key Features
Dual-Path Processing: * Fast Path (95%): Digital PDFs are parsed via PyMuPDF and classified by Mistral Small 3 in milliseconds.

Vision Fallback (5%): Scanned/Image-based PDFs trigger Llama 3.2 Vision 11B for visual classification.

European Context: Optimized for Spanish and Catalan document structures (e.g., Modelo 111, Nóminas, Contractes de treball) using Mistral's superior European language tuning.

Tag Memory (ChromaDB): Prevents "Tag Drift" by remembering previously used categories (e.g., ensuring "PAYSLIP" isn't renamed to "SALARY" the next day).

HPC Optimized: Configured to stay resident in the 48GB VRAM of an NVIDIA RTX 6000 Ada for zero-latency inference.

🏗 Architecture
The system is built with a decoupled, "Micro-Service" style architecture in Python:

reader.py: High-speed text extraction and "Vision Trigger" logic.

classifier.py: The brain; handles ChromaDB lookups and Ollama API orchestration.

filer.py: Safe I/O operations (Renaming, Path sanitization, and Moving).

config.py: Centralized hardware and logic thresholds.

main.py: CLI and orchestration layer.

🛠 Hardware Prerequisites
GPU: NVIDIA RTX 6000 Ada (48GB VRAM recommended for dual-model residency).

RAM: 256GB System RAM (allows for 90B model offloading if needed).

Environment: Windows 11 / Linux with Ollama installed.

🚦 Getting Started
1. Model Preparation
   Pull the specialized models via Ollama:

Bash
# High-precision European text model
ollama pull mistral-small:24b

# Multimodal fallback for scanned docs
ollama pull llama3.2-vision:11b
2. Configuration
   Update p2_config.py with your local paths:

Python
INPUT_FOLDER  = r"C:\Project2\input"
OUTPUT_ROOT   = r"C:\Project2\filed"
OLLAMA_MODEL_TEXT = "mistral-small:24b"
3. Usage
   Process a single file with debug logs:

PowerShell
python p2_main.py --pdf .\input\factura_einatec.pdf --debug
Run a "Dry Run" on a full folder (No files moved):

PowerShell
python p2_main.py --folder .\input\daily_batch --dry-run
🧠 Why Mistral Small 3?
While Llama 3.1 is powerful, Mistral Small 3 (24B) was selected for Project 2 priority due to:

Linguistic Nuance: Superior understanding of Spanish/Catalan legal and financial terminology.

Efficiency: At 24B parameters, it fits perfectly alongside the 11B Vision model in 48GB VRAM, allowing both to stay "Warm" (Resident) for instant processing.

Instruction Following: More consistent JSON output for automated backend filing compared to smaller 8B models.

📂 Filing Structure Logic
Files are renamed using the following convention:
{TAG}_{IDENTIFIER}_{ORIGINAL_NAME}.pdf

Example:

Input: document_001.pdf (A Spanish water bill)

Output: INVOICE_Agbar_document_001.pdf → Moved to /filed/INVOICE/

⚖️ License & Audit
Every action is logged to filing_log.jsonl for auditability, ensuring a clear record of how the AI moved each document.