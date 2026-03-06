# 🎭 Deepfake Audio-Visual Detection API

A temporal deepfake detection system that localizes **which seconds of a video are fake** by measuring the mismatch between lip movements and spoken phonemes using Dynamic Time Warping (DTW).

> **Novel contribution:** Instead of a binary "fake or real" verdict, this system outputs a per-second forgery score — e.g. `[2s–5s: 87% fake, 6s–9s: 12% fake]` — making it the first interpretable, training-free temporal localization pipeline for audio-visual deepfakes.

---

## 📌 How It Works

```
Video Input
    │
    ├──► MediaPipe Face Mesh ──► Lip Aperture Curve (per frame)
    │
    ├──► ffmpeg audio extract ──► Whisper ASR ──► Phoneme Activity Curve (per frame)
    │
    └──► Sliding Window DTW ──► Per-second Mismatch Scores
                                        │
                                        └──► Random Forest Classifier
                                                    │
                                                    └──► Prediction + Fake Segments
```

The core insight: in real videos, lip movements tightly follow phoneme activity. In deepfakes (especially Wav2Lip-generated), this synchronization breaks down at specific moments — creating measurable DTW distance spikes.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Accuracy | 71.67% |
| F1 Score (macro) | 0.713 |
| Precision | 0.729 |
| Recall | 0.717 |
| AUC-ROC | 0.742 |
| PR-AUC | 0.794 |
| Real video accuracy | 83.3% |
| Fake video accuracy | 60.0% |

Trained and evaluated on **200 videos** from the [FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb) dataset (100 real, 100 fake FVRA).

### Top discriminating features
1. `peak_ratio` — lip opening frequency vs phoneme peak frequency
2. `mismatch_closed_open` — frames where lips are closed but audio is open
3. `mad_global` — mean absolute deviation (fake lips are smoother than real ones)

---

## 🚀 API Endpoints

### `GET /`
Health check and endpoint listing.

### `GET /health`
Returns loaded model status.

### `POST /detect/upload`
Upload a video file directly.

**Request:**
```bash
curl -X POST "https://your-api.onrender.com/detect/upload" \
  -F "file=@your_video.mp4"
```

**Response:**
```json
{
  "filename": "video.mp4",
  "size_mb": 4.2,
  "prediction": "fake",
  "confidence": 0.82,
  "fake_ratio": 0.90,
  "fake_segments": [
    {"start": 0.48, "end": 4.84, "score": 0.503}
  ],
  "transcript": "important where we are in the world...",
  "features": {
    "mad_global": 0.29,
    "peak_ratio": 3.8,
    "pearson": 0.01
  }
}
```

### `POST /detect/url`
Pass a URL pointing to a video file.

**Request:**
```bash
curl -X POST "https://your-api.onrender.com/detect/url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/video.mp4"}'
```

**Response:** Same format as `/detect/upload`.

### Limits
- Maximum file size: **50 MB**
- Supported formats: `mp4`, `avi`, `mov`, `mkv`, `webm`

---

## 🛠️ Local Setup

### Prerequisites
- Python 3.11
- ffmpeg installed on your system
- espeak-ng installed on your system

### Install

```bash
git clone https://github.com/YOUR_USERNAME/deepfake-api.git
cd deepfake-api

pip install --upgrade pip setuptools wheel
pip install --no-build-isolation openai-whisper==20231117
pip install -r requirements.txt
```

### Run locally

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

---

## ☁️ Deployment (Render)

This repo is configured for one-click deployment on [Render](https://render.com) via `render.yaml`.

1. Fork or clone this repo
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` and configures everything
5. Click **Deploy**

Build time: ~5–10 minutes on first deploy.

---

## 📁 Project Structure

```
deepfake-api/
├── main.py                  # FastAPI app — routes and inference logic
├── pipeline.py              # Full ML pipeline (lip extraction, phonemes, DTW, features)
├── requirements.txt         # Python dependencies
├── render.yaml              # Render deployment config
├── rf_model.pkl             # Trained Random Forest classifier
├── scaler.pkl               # Feature scaler (StandardScaler)
├── face_landmarker.task     # MediaPipe face landmark model
└── README.md
```

---

## 🔬 Technical Details

### Lip Extraction
- MediaPipe Face Landmarker (478 landmarks)
- Lip aperture = distance(landmark 13, landmark 14) / face width
- Normalized by face width to handle varying camera distances
- Savitzky-Golay smoothing (window=11, polyorder=2)

### Phoneme Extraction
- Audio extracted at 16kHz mono via ffmpeg
- OpenAI Whisper `small` model with word-level timestamps
- Phonemes classified as open (vowels=1.0), partial (consonants=0.5), closed (bilabials=0.0)
- Distributed across frames by word timing

### DTW Sliding Window
- Window: 1.0 second, Step: 0.5 seconds (50% overlap)
- Score per window = mean(normalized DTW distance, inverted Pearson correlation)
- Fake segments = contiguous windows above calibrated threshold (0.2027)

### Classifier
- 18 handcrafted features per video
- Random Forest (100 trees)
- Trained on 140 videos, tested on 60

---

## 📚 Dataset

[FakeAVCeleb v1.2](https://github.com/DASH-Lab/FakeAVCeleb) — A multimodal deepfake dataset containing:
- `RealVideo-RealAudio` — ground truth real videos
- `FakeVideo-RealAudio` — face-swapped with original audio (primary test target)
- `FakeVideo-FakeAudio` — both video and audio manipulated
- `RealVideo-FakeAudio` — original video with swapped audio

---

## ⚠️ Limitations

- **False negative rate: 40%** — high-quality Wav2Lip fakes that produce natural lip dynamics can evade detection
- **Short videos only** — optimized for 3–6 second clips (FakeAVCeleb format)
- **English speech** — Whisper works cross-lingually but phoneme mapping is tuned for English
- **Front-facing faces** — MediaPipe accuracy degrades on profile or heavily occluded faces
- **No GPU required** — runs on CPU but inference takes 15–30 seconds per video

---

## 🗺️ Roadmap

- [ ] Expand training set to 1000+ videos
- [ ] Add FVFA (FakeVideo-FakeAudio) category support
- [ ] Cross-lingual phoneme mapping
- [ ] React frontend demo
- [ ] Async job queue for longer videos
- [ ] Confidence calibration

---

## 📄 License

MIT License. See `LICENSE` for details.

---

## 🙏 Acknowledgements

- [FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb) dataset by DASH Lab
- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [MediaPipe](https://github.com/google/mediapipe) for face landmark detection
- [dtaidistance](https://github.com/wannesm/dtaidistance) for DTW computation