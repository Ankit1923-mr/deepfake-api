import os
import sys
import uuid
import pickle
from pathlib import Path
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).parent
MODEL_PATH        = BASE_DIR / "face_landmarker.task"
RF_MODEL_PATH     = BASE_DIR / "rf_model.pkl"
SCALER_PATH       = BASE_DIR / "scaler.pkl"
OPTIMAL_THRESHOLD = 0.2027
MAX_FILE_MB       = 50
MAX_URL_MB        = 50

# ── Models container ─────────────────────────────────────────────
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")
    with open(RF_MODEL_PATH, "rb") as f:
        ml_models["rf"] = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        ml_models["scaler"] = pickle.load(f)
    ml_models["whisper"] = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"
    )
    print("All models loaded.")
    yield
    ml_models.clear()

app = FastAPI(
    title="Deepfake Audio-Visual Detection API",
    description="Detects deepfakes via lip-phoneme DTW alignment.",
    version="1.0.0",
    lifespan=lifespan
)

sys.path.insert(0, str(BASE_DIR))
from pipeline import run_pipeline

# ── Helpers ──────────────────────────────────────────────────────
def cleanup_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except:
        pass

def run_inference(video_path: str) -> dict:
    return run_pipeline(
        video_path        = video_path,
        model_path        = str(MODEL_PATH),
        rf_model          = ml_models["rf"],
        scaler            = ml_models["scaler"],
        whisper_model     = ml_models["whisper"],
        optimal_threshold = OPTIMAL_THRESHOLD
    )

# ── Routes ───────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Deepfake Detection API is running.",
        "endpoints": {
            "upload": "POST /detect/upload",
            "url":    "POST /detect/url",
            "health": "GET  /health"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": list(ml_models.keys())
    }

@app.post("/detect/upload")
async def detect_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload mp4, avi, mov, mkv, or webm."
        )

    contents = await file.read()
    size_mb = len(contents) / 1e6
    if size_mb > MAX_FILE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f} MB. Maximum allowed: {MAX_FILE_MB} MB."
        )

    suffix   = Path(file.filename).suffix
    tmp_path = f"/tmp/{uuid.uuid4().hex}{suffix}"
    with open(tmp_path, "wb") as f:
        f.write(contents)

    background_tasks.add_task(cleanup_file, tmp_path)

    try:
        result = run_inference(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return JSONResponse(content={
        "filename": file.filename,
        "size_mb":  round(size_mb, 2),
        **result
    })


class URLRequest(BaseModel):
    url: str

@app.post("/detect/url")
async def detect_url(
    request: URLRequest,
    background_tasks: BackgroundTasks
):
    url       = request.url
    path_part = url.split("?")[0].lower()

    if not any(path_part.endswith(ext) for ext in
               [".mp4", ".avi", ".mov", ".mkv", ".webm"]):
        raise HTTPException(
            status_code=400,
            detail="URL must point to a video file (mp4, avi, mov, mkv, webm)."
        )

    tmp_path = f"/tmp/{uuid.uuid4().hex}.mp4"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Could not download video. HTTP {response.status_code}"
                    )
                total     = 0
                max_bytes = MAX_URL_MB * 1_000_000
                with open(tmp_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        total += len(chunk)
                        if total > max_bytes:
                            cleanup_file(tmp_path)
                            raise HTTPException(
                                status_code=413,
                                detail=f"Video exceeds {MAX_URL_MB} MB limit."
                            )
                        f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        cleanup_file(tmp_path)
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")

    background_tasks.add_task(cleanup_file, tmp_path)

    try:
        result = run_inference(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return JSONResponse(content={
        "source_url": url,
        "size_mb":    round(os.path.getsize(tmp_path) / 1e6, 2),
        **result
    })