from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from paddleocr import PaddleOCRVL

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [ocr-api] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="OCR API", version="0.1.0")

Path(".tmp").mkdir(parents=True, exist_ok=True)

use_vllm = os.getenv("USE_VLLM", "0") == "1"
vllm_server_url = os.getenv("VLLM_SERVER_URL", "http://localhost:8000/v1")

paddleocr_kwargs = {
    "vl_rec_model_name": "PaddleOCR-VL-1.5-0.9B",
    "layout_detection_model_name": "PP-DocLayoutV3",
    "layout_detection_model_dir": "./models/PP-DocLayoutV3/",
    "layout_threshold": None,
    "layout_nms": None,
    "layout_unclip_ratio": None,
    "layout_merge_bboxes_mode": None,
    "vl_rec_model_dir": None,
    "vl_rec_backend": None,
    "vl_rec_server_url":  None,
    "vl_rec_max_concurrency": None,
    "vl_rec_api_model_name": None,
    "vl_rec_api_key": None,
    "doc_orientation_classify_model_name": None,
    "doc_orientation_classify_model_dir": None,
    "doc_unwarping_model_name": None,
    "doc_unwarping_model_dir": None,
    "use_doc_orientation_classify": None,
    "use_doc_unwarping": None,
    "use_layout_detection": None,
    "use_chart_recognition": None,
    "use_seal_recognition": None,
    "use_ocr_for_image_block": False,
    "format_block_content": None,
    "merge_layout_blocks": True,
    "markdown_ignore_labels": None,
    "use_queues": None
}

if use_vllm:
    paddleocr_kwargs.update(
        {
            "vl_rec_backend": "vllm-server",
            "vl_rec_server_url": vllm_server_url,
        }
    )
else:
    paddleocr_kwargs.update(
        {
            "vl_rec_model_dir": "./models/PaddleOCR-VL-1.5",
        }
    )

pipeline = PaddleOCRVL(**paddleocr_kwargs)

if use_vllm:
    logger.info("PaddleOCR initialized with vLLM backend at %s", vllm_server_url)
else:
    logger.info("PaddleOCR initialized with local backend model_dir=./models/PaddleOCR-VL-1.5")

@app.get("/health")
async def health() -> dict[str, str]:
    if use_vllm:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{vllm_server_url}/models")
                resp.raise_for_status()
        except Exception:
            raise HTTPException(status_code=503, detail="vLLM not ready")
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    logger.info("Received /predict request (filename=%s)", file.filename)
    results = []

    with TemporaryDirectory(dir=".tmp", delete=True) as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / file.filename
        input_path.write_bytes(await file.read())

        try:
            output = pipeline.predict(str(input_path))

        except Exception as exc:
            logger.exception("Prediction failed for filename=%s", file.filename)
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

        for idx, res in enumerate(output):
            json_path = tmp_path / f"output_{idx}.json"
            
            res.save_to_json(save_path=str(json_path))

            results.append(
                json.loads(
                    json_path.read_text(encoding="utf-8")
                )
            )
            
    return results