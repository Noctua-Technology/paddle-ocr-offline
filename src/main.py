import os
import tempfile
import numpy as np
from PIL import Image
from io import BytesIO
from paddleocr import PaddleOCR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(BASE_DIR, "local_models")

class PaddleOCRService:
    def __init__(self):
        self.models = {}

        # Base path for models (configurable via env var for Cloud Run)
        self.models_base = os.environ.get("MODELS_BASE", "./src/local_models")

        # Detection model (shared across languages)
        self.det_model = "PP-OCRv5_server_det"

        # Paddle map to names of rec files/modes
        self.model_registry = {
            "ar": "arabic_PP-OCRv5_mobile_rec",
            "en": "en_PP-OCRv5_mobile_rec",
            "fa": "arabic_PP-OCRv5_mobile_rec",
            "fr": "latin_PP-OCRv5_mobile_rec",
            "de": "latin_PP-OCRv5_mobile_rec",
            "hi": "devanagari_PP-OCRv3_mobile_rec",
            "id": "latin_PP-OCRv5_mobile_rec",
            "ja": "japan_PP-OCRv3_mobile_rec",
            "ko": "korean_PP-OCRv5_mobile_rec",
            "zh": "PP-OCRv5_mobile_rec",
            "pt": "latin_PP-OCRv5_mobile_rec",
            "ru": "cyrillic_PP-OCRv3_mobile_rec",
            "es": "latin_PP-OCRv5_mobile_rec",
            "ur": "arabic_PP-OCRv5_mobile_rec",
            "vi": "latin_PP-OCRv5_mobile_rec",
            "so": "en_PP-OCRv5_mobile_rec",
            "tl": "en_PP-OCRv5_mobile_rec",
        }

        # Paddle lang code to PaddleOCR lang code
        self.paddle_lang_map = {
            "ar": "ar",
            "en": "en",
            "fa": "fa",
            "fr": "fr",
            "de": "german",
            "hi": "hi",
            "id": "id",
            "ja": "japan",
            "ko": "korean",
            "zh": "ch",
            "pt": "pt",
            "ru": "ru",
            "es": "es",
            "ur": "ur",
            "vi": "vi",
            "so": "en",
            "tl": "en",
        }

    def _get_or_load_model(self, lang: str):
        """Loads the model into memory with path validation."""
        if lang not in self.models:
            paddle_code = self.paddle_lang_map.get(lang, "en")
            rec_model = self.model_registry.get(lang, "en_PP-OCRv5_mobile_rec")

            # Note: Assuming this initializes a PaddleX pipeline that supports .predict()
            self.models[lang] = PaddleOCR(
                det_model_dir=f"{self.models_base}/{self.det_model}",
                rec_model_dir=f"{self.models_base}/{rec_model}",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang=paddle_code,
            )
            
        return self.models[lang]

    def run_ocr(self, file_bytes: bytes, content_type: str, lang: str = "en"):
        ocr = self._get_or_load_model(lang)
        temp_pdf_path = None
        input_data = None
        common_types = {
            "application/pdf": ".pdf",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
        }
        ext = common_types.get(content_type)
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
                f.write(file_bytes)
                temp_pdf_path = f.name
                input_data = temp_pdf_path
                results = ocr.predict(input=input_data)
                
            # normalize results to a list (Image input returns object, PDF input returns list)
            if not isinstance(results, list):
                results = [results]

            text_boxes = []
            all_text_lines = []

            for page_idx, page_result in enumerate(results):
                texts = page_result.get("rec_texts", [])
                boxes = page_result.get("dt_polys", [])

                for text, box in zip(texts, boxes):
                    all_text_lines.append(text)
                    text_boxes.append({
                        "text": text,
                        "box": box.tolist() if hasattr(box, "tolist") else box,
                        "page": page_idx + 1
                    })

            return {
                "full_text": " ".join(all_text_lines), 
                "text_boxes": text_boxes
            }
        finally:
            # REQUIRED TO CLEAN UP TEMP FILES
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

ocr_service = PaddleOCRService()