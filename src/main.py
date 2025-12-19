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
        """Loads the model into memory with path validation."""
        if lang not in self.models:
            paddle_code = self.model_registry.get(lang, "en")
            
            lang_folder = os.path.join(MODELS_ROOT, paddle_code)
            rec_path = os.path.join(lang_folder, "rec")
            det_path = os.path.join(lang_folder, "det")

            # Find the correct rec_model path
            final_rec_path = rec_path if os.path.exists(rec_path) else lang_folder
            
            model_file_check = os.path.join(final_rec_path, "inference.pdiparams")
            
            if not os.path.exists(model_file_check):
                raise FileNotFoundError(
                    f"OCR Model not found. Checked: {rec_path} and {lang_folder}. "
                    f"Ensure 'inference.pdiparams' exists in one of these locations."
                )

            self.models[lang] = PaddleOCR(
                det_model_dir=det_path if os.path.exists(det_path) else lang_folder,
                rec_model_dir=final_rec_path,
                use_angle_cls=False,
                lang=paddle_code,
                device='cpu'
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