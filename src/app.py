import os
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from typing import Literal
from main import ocr_service
import numpy as np
from PIL import Image
from io import BytesIO



app = FastAPI(title="Paddle OCR")

SupportedLangs = Literal["ar", "en", "fr", "de", "ko", "zh", "ja", "es", "hi"]

@app.get("/")
async def root():
    return {"message": "OCR API is online and ready"}

@app.get("/health")
def health_check():
    try:
        if not ocr_service.models:
             ocr_service._get_or_load_model("en")
        
        return {
            "status": "ready",
            "models_loaded": list(ocr_service.models.keys()),
            "message": "OCR engine is readu"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"OCR not ready: {str(e)}")

@app.post("/ocr/predict")
async def predict_text(
    file: UploadFile = File(...),
    lang: SupportedLangs = Query("en")
):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        img_array = np.array(image)

        return ocr_service.run_ocr(img_array, lang=lang)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR Error: {str(e)}")
    


