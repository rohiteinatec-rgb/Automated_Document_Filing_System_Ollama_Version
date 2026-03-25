import io
import base64
import fitz
from PIL import Image, ImageEnhance
from config import Config

class PDFHandler:
    @staticmethod
    def is_text_based(pdf_path: str) -> bool:
        """Analyzes PDF geometry to route extraction."""
        try:
            doc = fitz.open(pdf_path)
            page = doc[0]
            text_area, image_area = 0.0, 0.0

            for block in page.get_text("blocks"):
                area = abs(fitz.Rect(block[:4]))
                if block[6] == 0: text_area += area
                elif block[6] == 1: image_area += area
            doc.close()

            return text_area > image_area
        except Exception:
            return False

    @staticmethod
    def to_base64_images(pdf_path: str, dpi: int = Config.RENDER_DPI) -> list:
        """Renders PDF pages to enhanced base64 images."""
        images_b64 = []
        try:
            doc = fitz.open(pdf_path)
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            for page in doc:
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Enhancement
                img = ImageEnhance.Sharpness(img).enhance(2.0)
                img = ImageEnhance.Contrast(img).enhance(1.8)

                buf = io.BytesIO()
                img.save(buf, format="PNG")
                images_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
            doc.close()
        except Exception as e:
            print(f"    [ImageRender] Failed: {e}")
        return images_b64