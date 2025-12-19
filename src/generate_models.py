import os
import shutil
from paddleocr import PaddleOCR

LANG_MAP = {
    "Arabic": "ar",
    "English": "en",
    "Farsi": "fa",
    "French": "fr",
    "German": "german",
    "Hindi": "hi",
    "Indonesian": "id",
    "Japanese": "japan",
    "Korean": "korean",
    "Mandarin Chinese": "ch",
    "Portuguese": "pt",
    "Russian": "ru",
    "Somali": "en", # PaddleOCR does not have a model for Somali
    "Spanish": "es",
    "Tagalog": "en", # PaddleOCR does not have a model for Tagalog
    "Urdu": "ur",
    "Vietnamese": "vi"
}

def download_models():
    for name, code in LANG_MAP.items():
        
        try:
            ocr = PaddleOCR(lang=code, use_gpu=False)
            
            try:
                src_path = ocr.paddlex_pipeline.text_rec_model.model_dir
            except AttributeError:
                src_path = ocr.page_pipeline.text_rec.model_dir

            dest = f"./local_models/{code}/rec"
            
            if os.path.exists(src_path):
                shutil.copytree(src_path, dest, dirs_exist_ok=True)
                print(f"Success: {name} saved to {dest}")
            else:
                print(f"Error: Could not find source path for {name}")
                
        except Exception as e:
            print(f"Skipping {name}: {e}")

if __name__ == "__main__":
    download_models()