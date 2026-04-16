import os
import sys
import uuid
import pickle
from pathlib import Path
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR           = Path(__file__).parent
MODEL_PATH         = BASE_DIR / "face_landmarker.task"
RF_MODEL_PATH      = BASE_DIR / "rf_model.pkl"
SCALER_PATH        = BASE_DIR / "scaler.pkl"

# ── Thresholds ───────────────────────────────────────────────────
RF_PROBA_THRESHOLD = 0.38  # Optimal ML threshold for classifying video as fake
DTW_SEG_THRESHOLD  = 0.35    # Signal threshold for highlighting specific fake segments
MAX_FILE_MB        = 50
MAX_URL_MB         = 50

# ── Models container ─────────────────────────────────────────────
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")
    with open(RF_MODEL_PATH, "rb") as f:
        ml_models["rf"] = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        ml_models["scaler"] = pickle.load(f)
    # Groq client — no model loaded in RAM
    from groq import Groq
    ml_models["groq"] = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    print("All models loaded.")
    yield
    ml_models.clear()

app = FastAPI(
    title="Deepfake Audio-Visual Detection API",
    description="Detects deepfakes via lip-phoneme DTW alignment.",
    version="1.0.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        video_path         = video_path,
        model_path         = str(MODEL_PATH),
        rf_model           = ml_models["rf"],
        scaler             = ml_models["scaler"],
        groq_client        = ml_models["groq"],
        rf_proba_thresh    = RF_PROBA_THRESHOLD,
        dtw_seg_thresh     = DTW_SEG_THRESHOLD
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
    if not file.filename.lower().endswith(
            (".mp4", ".avi", ".mov", ".mkv", ".webm")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type."
        )
    contents = await file.read()
    size_mb  = len(contents) / 1e6
    if size_mb > MAX_FILE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f} MB. Max: {MAX_FILE_MB} MB."
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
    url      = request.url
    tmp_path = f"/tmp/{uuid.uuid4().hex}.mp4"
    path_part = url.split("?")[0].lower()

    is_direct = any(path_part.endswith(ext) for ext in
                    [".mp4", ".avi", ".mov", ".mkv", ".webm"])

    try:
        if is_direct:
            # Direct video file — download via httpx
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
        else:
            # YouTube / social media URL — download via yt-dlp
            import yt_dlp
            ydl_opts = {
                "format":    "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "outtmpl":   tmp_path,
                "quiet":     True,
                "no_warnings": True,
                "merge_output_format": "mp4",
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            if not os.path.exists(tmp_path):
                # yt-dlp sometimes adds extension even if outtmpl has .mp4
                guessed = tmp_path.replace(".mp4", "") + ".mp4"
                if os.path.exists(guessed):
                    tmp_path = guessed
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="yt-dlp could not download the video."
                    )

    except HTTPException:
        raise
    except Exception as e:
        cleanup_file(tmp_path)
        raise HTTPException(
            status_code=400, detail=f"Download failed: {str(e)}"
        )
    
    # Run the ML pipeline
    background_tasks.add_task(cleanup_file, tmp_path)
    try:
        result = run_inference(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
        
    return JSONResponse(content={
        "source_url": url,
        **result
    })
        
@app.post("/debug/scores")
async def debug_scores(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    contents = await file.read()
    tmp_path = f"/tmp/{uuid.uuid4().hex}.mp4"
    with open(tmp_path, "wb") as f:
        f.write(contents)
    background_tasks.add_task(cleanup_file, tmp_path)
    from pipeline import extract_lip_curve, extract_audio, transcribe_audio, words_to_phoneme_curve, sliding_window_dtw_scores_v2, load_landmarker
    import tempfile
    with load_landmarker(str(MODEL_PATH)) as lm:
        lip_curve, fps, fc = extract_lip_curve(tmp_path, lm)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    extract_audio(tmp_path, tmp_wav)
    result = transcribe_audio(tmp_wav, ml_models["groq"])
    ph = words_to_phoneme_curve(result, fc, fps)
    scores = sliding_window_dtw_scores_v2(lip_curve, ph, fps)
    return {"window_scores": scores, "fps": fps}