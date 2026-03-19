# Smart PDF Text Extractor (OCRmyPDF Version)

This pipeline extracts raw text from PDF documents. It uses a "smart routing" approach to dynamically choose the fastest and most accurate extraction method based on the document's composition.

## 🚀 How It Works (The Routing Logic)

Instead of blindly running OCR on every document, this script analyzes the geometric area of the first page:
1. **Digital PDF (Fast Track):** If the page consists primarily of embedded text blocks, it bypasses OCR entirely and instantly extracts the native text using PyMuPDF.
2. **Scanned PDF (OCR Track):** If the page consists primarily of images, it spins up an isolated Docker container running `ocrmypdf` (Tesseract) to perform Optical Character Recognition.

## 📋 Prerequisites

Before running the script, ensure you have the following installed:
* **Python 3.8+**
* **Docker:** Must be installed and running on your machine (required for the isolated OCR engine).

### Python Dependencies
Install the required Python library for the geometry analysis:
```bash
pip install PyMuPDF