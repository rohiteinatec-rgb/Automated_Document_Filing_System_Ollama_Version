📄 Spain-Optimized PDF Extractor with Self-Healing Vision AI
An advanced, multi-track document extraction pipeline designed specifically for Spanish invoices. This tool dynamically analyzes incoming PDFs to determine if they are digital or scanned, routing them through specialized extraction tracks. It features a rigorous "Quality Gate" that checks for Spanish financial compliance (IBANs, CIF/NIEs) and automatically escalates failed or corrupted OCR scans to a local instance of Llama 3.2 Vision for self-healing and correction.

✨ Key Features
Intelligent Routing: Automatically detects whether a PDF is natively digital (text-based) or scanned (image-based) by analyzing the document's geometric blocks, routing it to the most efficient pipeline.

Tiered Digital Track (Track A): Processes digital PDFs using a blazing-fast escalation path: PyMuPDF4LLM → pdfplumber → Docling (Digital Mode).

Self-Healing Scanned Track (Track B): Processes scanned PDFs using Docling's EasyOCR. If the output is garbage, it automatically escalates to a local Ollama Vision AI to read and correct the document from scratch.

Strict Quality Gate: A built-in evaluation function that grades the extraction based on:

Generic health (character count, garbage ratio, font corruption).

OCR Hallucinations (letters inside numbers, mangled percentage signs).

Table integrity (missing values next to "IVA" or "Base Imponible").

Semantic presence of Spanish data (CIF/NIE, ES-based IBANs, Euro amounts).

100% Local & Private: All OCR and LLM vision inference runs locally. No sensitive invoice data is sent to external cloud APIs.

⚙️ Architecture
Plaintext
Incoming PDF
│
├─> [Gatekeeper] Is it Digital or Scanned?
│
├─> [Track A: Digital]
│      ├─> Tier 1: PyMuPDF4LLM
│      ├─> Tier 2: pdfplumber (If Tier 1 fails Quality Gate)
│      └─> Tier 3: Docling (If Tier 2 fails Quality Gate)
│
└─> [Track B: Scanned]
├─> Tier 1: Docling + EasyOCR
└─> Tier 2: Ollama Vision AI (If OCR fails Quality Gate)
🛠️ Prerequisites
Before running the script, ensure you have the following installed on your system:

Python 3.9+

Ollama: Must be installed and running locally.

Llama 3.2 Vision Model: You need to pull the vision model into your local Ollama instance:

Bash
ollama run llama3.2-vision:latest
Python Dependencies
Install the required Python packages:

Bash
pip install pymupdf pymupdf4llm pdfplumber Pillow docling requests
🚀 Usage
Run the script from your terminal, pointing it to the PDF you want to extract:

Bash
python detector_ollama.py --pdf path/to/your/invoice.pdf

Examples
Run with debug mode (Recommended for testing):

Bash
python detector_ollama.py --pdf input/sc_01_automocion.pdf --debug
Run with specific OCR languages:

Bash
python detector_ollama.py --pdf input/invoice_catalan.pdf --lang ca+es+en
📂 Output
The script generates a flawlessly formatted Markdown (.txt) file representing the extracted invoice, including reconstructed tables and recovered footers.

Outputs are automatically saved in the output/ directory relative to where the script is executed, using the same filename as the original PDF:
output/invoice_name.txt

🧠 Under the Hood: The "Anti-Lazy" Vision Prompt
If EasyOCR fails the Quality Gate, the script uses the Pillow library to drastically enhance the contrast and sharpness of the PDF pages. It then feeds both the enhanced images and the flawed OCR text into llama3.2-vision.

The LLM is prompted with strict instructions to fix corrupted numbers (e.g., 6.7oo.00 → 6.700,00), rebuild empty tables, and forcibly extract the tiny footer text (IBANs/Registry details) that standard OCR engines frequently miss on Spanish invoices.