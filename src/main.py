import os
from paddleocr import PaddleOCR
from fastapi import File


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

            self.models[lang] = PaddleOCR(
                det_model_dir=f"{self.models_base}/{self.det_model}",
                rec_model_dir=f"{self.models_base}/{rec_model}",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang=paddle_code,
            )
            
        return self.models[lang]

    def run_ocr(self, file, lang: str = "en"):
        """Executes OCR and returns structured JSON-ready data"""
        ocr_engine = self._get_or_load_model(lang)
        result = ocr_engine.predict(input=file)

        all_text = []
        text_boxes = []

        for res in result:
            texts = res.get("rec_texts", [])
            boxes = res.get("dt_polys", [])

            for text, box in zip(texts, boxes):
                all_text.append(text)

                clean_box = box.tolist() if hasattr(box, "tolist") else box

                text_boxes.append({"text": text, "box": clean_box})

        return {"full_text": " ".join(all_text), "text_boxes": text_boxes}

ocr_service = PaddleOCRService()
