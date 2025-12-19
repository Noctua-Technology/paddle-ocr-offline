import os
from paddleocr import PaddleOCR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(BASE_DIR, 'local_models')

class PaddleOCRService:
    def __init__(self):
        self.models = {}
        
        self.model_registry = {
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
            "tl": "en" 
        }

    def _get_or_load_model(self, lang: str):
        """Loads the model into memory without recursion."""
        if lang not in self.models:
            # defaults to en if lang not found
            paddle_code = self.model_registry.get(lang)
            
            if not paddle_code:
                paddle_code = "en"
            
            rec_path = os.path.join(MODELS_ROOT, paddle_code, "rec")
            det_path = os.path.join(MODELS_ROOT, paddle_code, "det")

            if not os.path.exists(rec_path):
                raise FileNotFoundError(f"Model path missing at {rec_path}. ")

            
            self.models[lang] = PaddleOCR(
                det_model_dir=det_path if os.path.exists(det_path) else None,
                rec_model_dir=rec_path,
                use_angle_cls=False,
                lang=paddle_code,
                use_gpu=False,
                show_log=False
            )
        return self.models[lang]

    def run_ocr(self, image_path: str, lang: str = 'en'):
        """Executes OCR and returns structured JSON-ready data"""
        ocr_engine = self._get_or_load_model(lang)
        result = ocr_engine.predict(input=image_path)

        all_text = []
        text_boxes = []

        for res in result:
            texts = res.get('rec_texts', [])
            boxes = res.get('dt_polys', [])

            for text, box in zip(texts, boxes):
                all_text.append(text)
                
                clean_box = box.tolist() if hasattr(box, 'tolist') else box
                
                text_boxes.append({
                    "text": text,
                    "box": clean_box
                })
        
        return {
            "full_text": " ".join(all_text),
            "text_boxes": text_boxes
        }

ocr_service = PaddleOCRService()