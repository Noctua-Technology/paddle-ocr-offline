import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from typing import Literal
from main import ocr_service

app = FastAPI(title="Paddle OCR")

SupportedLangs = Literal["ar", "en", "fr", "de", "ko", "zh", "ja", "es", "hi"]

@app.get("/")
async def root():
    return {"message": "OCR API is online and ready"}

@app.post("/ocr/predict")
async def predict_text(
    file: UploadFile = File(...),
    lang: SupportedLangs = Query("en")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    temp_file_path = os.path.join(base_dir, f"temp_{file.filename}")

    try:
        content = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(content)

        ocr_result = ocr_service.run_ocr(temp_file_path, lang=lang)

        return {
            "filename": file.filename,
            "language": lang,
            "extracted_text": ocr_result["full_text"],
            "text_boxes": ocr_result["text_boxes"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR Error: {str(e)}")
    
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)