import os
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from typing import Literal
from .main import ocr_service
import numpy as np
from PIL import Image
from io import BytesIO
from pdf2image import convert_from_bytes



app = FastAPI(title="Paddle OCR")

SupportedLangs = Literal["ar", "en", "fa", "fr", "de", "hi", "id", "ja", "ko", "zh", "pt", "ru", "es", "ur", "vi"]

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
            "message": "OCR engine is ready"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"OCR not ready: {str(e)}")

@app.post("/ocr/predict")
async def predict_text(
    file: UploadFile = File(...),
    lang: SupportedLangs = Query("en")
):
    try:
        file_bytes = await file.read()

        # Check if it's a PDF
        is_pdf = (file.content_type == "application/pdf") or \
                 (file.filename.lower().endswith(".pdf"))

        if is_pdf:
            # Convert PDF bytes to a list of PDF Images (one per page)
            try:
                images = convert_from_bytes(file_bytes)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"PDF Conversion Failed.Error: {e}")
            
            # Run OCR on every page and collect results
            results = []
            for i, img in enumerate(images):
                img_array = np.array(img)
                text = ocr_service.run_ocr(img_array, lang=lang)
                results.append(f"--- Page {i+1} ---\n{text}")
            
            return "\n".join(results)

        else:
            image_bytes = await file.read()
            image = Image.open(BytesIO(image_bytes))
            img_array = np.array(image)
            return ocr_service.run_ocr(img_array, lang=lang)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR Error: {str(e)}")
    


