import os
import shutil
from paddleocr import PaddleOCR

LANG = 'en'
DEST = f"./local_models/{LANG}/rec"


def function_name(lang = LANG):
    ocr = PaddleOCR(lang=lang)

    try:
        src_path = ocr.paddlex_pipeline.text_rec_model.model_dir
    except AttributeError:
        src_path = ocr.page_pipeline.text_rec.model_dir

    if os.path.exists(src_path):
        shutil.copytree(src_path, DEST, dirs_exist_ok=True)
        print(f"Success: {LANG} rec model saved to {DEST}")
    else:
        print(f"Error: Could not find source path {src_path}")