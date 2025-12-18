import os
from paddleocr import PaddleOCR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(BASE_DIR, 'local_models')
OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

class PaddleOCRService:
    def __init__(self):
        self.models = {}
        
        self.model_registry = {
            "ar": {
                "det": "PP-OCRv5_server_det",
                "rec": "arabic"
            },
            "en": {
                "det": "PP-OCRv5_server_det", 
                "rec": "en"
            },
            "fr": {
                "det": "PP-OCRv5_server_det",
                "rec": "fr/rec"
            },
            "de": {
                "det": "PP-OCRv5_server_det",
                "rec": "german/rec"
            },
            "ko": {
                "det": "PP-OCRv5_server_det",
                "rec": "korean"
            },
            "zh": {
                "det": "PP-OCRv5_server_det",
                "rec": "zh"
            },
            "ja": {
                "det": "PP-OCRv5_server_det",
                "rec": "japanese"
            },
            "es": {
                "det": "PP-OCRv5_server_det",
                "rec": "es"
            },
            "hi": {
                "det": "PP-OCRv5_server_det",
                "rec": "hindi"
            }
        }

    def _get_or_load_model(self, lang: str):
        """Loads the model into memory only if it hasn't been loaded yet"""
        if lang not in self.models:
            if lang not in self.model_registry:
                print(f"Language {lang} not found, falling back to English")
                return self._get_or_load_model("en")

            config = self.model_registry[lang]
            det_path = os.path.join(MODELS_ROOT, config["det"])
            rec_path = os.path.join(MODELS_ROOT, config["rec"])

            if not os.path.exists(det_path) or not os.path.exists(rec_path):
                raise FileNotFoundError(f"Model folders for {lang} missing in {MODELS_ROOT}")

            print(f"Loading {lang} model into memory")
            
            self.models[lang] = PaddleOCR(
                det_model_dir=det_path,
                rec_model_dir=rec_path,
                use_angle_cls=True,
                lang=lang,
                device='cpu'
            )
        return self.models[lang]

    def run_ocr(self, image_path: str, lang: str = 'ar'):
        """Executes OCR and outputs the result"""
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