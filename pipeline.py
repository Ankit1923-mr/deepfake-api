import cv2
import numpy as np
import subprocess
import os
import tempfile

from scipy.signal import savgol_filter, find_peaks
from scipy.stats import pearsonr
from scipy.signal import correlate

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ── MediaPipe setup ──────────────────────────────────────────────
def load_landmarker(model_path):
    BaseOptions          = mp_python.BaseOptions
    FaceLandmarker       = mp_vision.FaceLandmarker
    FaceLandmarkerOptions = mp_vision.FaceLandmarkerOptions
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        num_faces=1
    )
    return FaceLandmarker.create_from_options(options)


def get_lip_aperture(frame_rgb, landmarker):
    h, w    = frame_rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result  = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    lms  = result.face_landmarks[0]
    p13  = np.array([lms[13].x * w,  lms[13].y * h])
    p14  = np.array([lms[14].x * w,  lms[14].y * h])
    p78  = np.array([lms[78].x * w,  lms[78].y * h])
    p308 = np.array([lms[308].x * w, lms[308].y * h])
    lip_aperture = np.linalg.norm(p13 - p14)
    face_width   = np.linalg.norm(p78 - p308)
    if face_width == 0:
        return None
    return lip_aperture / face_width


def extract_lip_curve(video_path, landmarker):
    cap         = cv2.VideoCapture(video_path)
    fps         = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lip_values  = []
    last_valid  = 0.0
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            lip_values.append(last_valid)
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        val = get_lip_aperture(frame_rgb, landmarker)
        if val is None:
            lip_values.append(last_valid)
        else:
            last_valid = val
            lip_values.append(val)
    cap.release()
    return np.array(lip_values, dtype=np.float32), fps, frame_count


# ── Signal processing ────────────────────────────────────────────
def smooth_lip_curve(curve):
    window = min(11, len(curve) if len(curve) % 2 == 1 else len(curve) - 1)
    if window < 3:
        return curve
    smoothed = savgol_filter(curve, window_length=window, polyorder=2)
    mn, mx   = smoothed.min(), smoothed.max()
    if mx - mn == 0:
        return smoothed
    return (smoothed - mn) / (mx - mn)


# ── Audio extraction ─────────────────────────────────────────────
def extract_audio(video_path, out_path):
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", "16000", "-vn", out_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode


# ── Faster-Whisper transcription ─────────────────────────────────
def transcribe_audio(wav_path, whisper_model):
    """
    Transcribe using faster-whisper and return output in the same
    format as the original openai-whisper so the rest of the pipeline
    needs no changes.
    """
    segments, info = whisper_model.transcribe(
        wav_path,
        word_timestamps=True,
        language="en"
    )
    result_segments = []
    full_text       = ""
    for segment in segments:
        words = []
        for word in segment.words:
            words.append({
                "word":  word.word,
                "start": word.start,
                "end":   word.end
            })
        result_segments.append({
            "text":  segment.text,
            "words": words
        })
        full_text += segment.text
    return {
        "text":     full_text.strip(),
        "segments": result_segments
    }


# ── Phoneme conversion ───────────────────────────────────────────
def word_to_openness(word):
    try:
        from phonemizer import phonemize
        phon = phonemize(
            word.strip().lower(),
            backend="espeak",
            language="en-us",
            with_stress=False,
            preserve_punctuation=False
        )
        seq = []
        for ch in phon.strip():
            if ch in "aeiouæɑɔəɛɪʊʌ":
                seq.append(1.0)
            elif ch in "pbm":
                seq.append(0.0)
            elif ch == " ":
                continue
            else:
                seq.append(0.5)
        return seq if seq else [0.5]
    except:
        return [0.5]


def words_to_phoneme_curve(whisper_result, total_frames, fps):
    curve = np.zeros(total_frames, dtype=np.float32)
    for segment in whisper_result["segments"]:
        for word_info in segment.get("words", []):
            word    = word_info["word"]
            t_start = word_info["start"]
            t_end   = word_info["end"]
            f_start = max(0, min(int(t_start * fps), total_frames - 1))
            f_end   = max(0, min(int(t_end   * fps), total_frames - 1))
            if f_start >= f_end:
                f_end = f_start + 1
            openness_seq = word_to_openness(word)
            n_frames     = f_end - f_start
            for j in range(n_frames):
                idx = min(int(j / n_frames * len(openness_seq)),
                          len(openness_seq) - 1)
                curve[f_start + j] = openness_seq[idx]
    return curve


# ── DTW sliding window ───────────────────────────────────────────
def sliding_window_dtw_scores_v2(lip_curve, phoneme_curve, fps,
                                  window_sec=1.0, step_sec=0.5):
    from dtaidistance import dtw
    window_frames = int(window_sec * fps)
    step_frames   = int(step_sec   * fps)
    n_frames      = min(len(lip_curve), len(phoneme_curve))
    lip_s         = smooth_lip_curve(lip_curve).astype(np.double)
    ph            = phoneme_curve.astype(np.double)
    scores        = []
    pos           = 0
    while pos + window_frames <= n_frames:
        lip_win = lip_s[pos:pos + window_frames]
        ph_win  = ph  [pos:pos + window_frames]
        dtw_score = dtw.distance(lip_win, ph_win) / window_frames
        if lip_win.std() < 1e-6 or ph_win.std() < 1e-6:
            corr_score = 0.5
        else:
            r, _       = pearsonr(lip_win, ph_win)
            corr_score = 1.0 - abs(r)
        scores.append({
            "start":      pos / fps,
            "end":        (pos + window_frames) / fps,
            "dtw_norm":   dtw_score,
            "corr_score": corr_score,
            "combined":   (dtw_score + corr_score) / 2.0
        })
        pos += step_frames
    return scores


def detect_fake_segments(window_scores, threshold=0.45):
    if not window_scores:
        return [], 0.0
    flagged = [
        (s["start"], s["end"], s["combined"])
        for s in window_scores if s["combined"] >= threshold
    ]
    if not flagged:
        return [], 0.0
    merged        = []
    cs, ce, sc    = flagged[0]
    for start, end, score in flagged[1:]:
        if start <= ce + 0.6:
            ce = max(ce, end)
            sc = (sc + score) / 2
        else:
            merged.append((cs, ce, sc))
            cs, ce, sc = start, end, score
    merged.append((cs, ce, sc))
    total_duration = window_scores[-1]["end"]
    fake_seconds   = sum(e - s for s, e, _ in merged)
    fake_ratio     = fake_seconds / total_duration if total_duration > 0 else 0.0
    return merged, fake_ratio


# ── Feature extraction ───────────────────────────────────────────
def extract_features(lip_curve, phoneme_curve, fps,
                     optimal_threshold=0.45):
    lip_s = smooth_lip_curve(lip_curve).astype(np.double)
    ph    = phoneme_curve.astype(np.double)
    n     = min(len(lip_s), len(ph))
    lip_s, ph = lip_s[:n], ph[:n]
    duration  = n / fps

    pearson_val = 0.0
    if lip_s.std() >= 1e-6 and ph.std() >= 1e-6:
        pearson_val, _ = pearsonr(lip_s, ph)

    xcorr      = correlate(lip_s - lip_s.mean(),
                           ph    - ph.mean(), mode="full")
    xcorr_peak = np.max(np.abs(xcorr)) / (
        n * (lip_s.std() * ph.std() + 1e-8)
    )

    lip_variance     = float(np.var(lip_s))
    lip_mad          = float(np.mean(np.abs(lip_s - lip_s.mean())))
    lip_range        = float(lip_s.max() - lip_s.min())
    phoneme_coverage = float(np.mean(ph > 0))

    lip_peaks, _ = find_peaks(lip_s, height=0.3,
                               distance=int(fps * 0.2))
    ph_peaks,  _ = find_peaks(ph,    height=0.4,
                               distance=int(fps * 0.2))
    lip_peak_rate = len(lip_peaks) / duration if duration > 0 else 0
    ph_peak_rate  = len(ph_peaks)  / duration if duration > 0 else 0
    peak_ratio    = float(np.clip(
        lip_peak_rate / (ph_peak_rate + 1e-8), 0, 50
    ))

    mad_global           = float(np.mean(np.abs(lip_s - ph)))
    mismatch_open_closed = float(np.mean((lip_s > 0.4) & (ph < 0.3)))
    mismatch_closed_open = float(np.mean((lip_s < 0.2) & (ph > 0.7)))

    window_scores = sliding_window_dtw_scores_v2(
        lip_curve, phoneme_curve, fps
    )
    if window_scores:
        combined  = [s["combined"]   for s in window_scores]
        dtw_s     = [s["dtw_norm"]   for s in window_scores]
        corr_s    = [s["corr_score"] for s in window_scores]
        mean_dtw      = float(np.mean(dtw_s))
        max_dtw       = float(np.max(dtw_s))
        std_dtw       = float(np.std(dtw_s))
        mean_corr     = float(np.mean(corr_s))
        mean_combined = float(np.mean(combined))
        std_combined  = float(np.std(combined))
        segs, fake_ratio = detect_fake_segments(
            window_scores, threshold=optimal_threshold
        )
        n_fake_segments = len(segs)
    else:
        mean_dtw = max_dtw = std_dtw = 0.0
        mean_corr = mean_combined = std_combined = 0.0
        fake_ratio      = 0.0
        n_fake_segments = 0

    return {
        "pearson":              float(pearson_val),
        "xcorr_peak":           float(xcorr_peak),
        "lip_variance":         lip_variance,
        "lip_mad":              lip_mad,
        "lip_range":            lip_range,
        "phoneme_coverage":     phoneme_coverage,
        "peak_ratio":           peak_ratio,
        "mad_global":           mad_global,
        "mismatch_open_closed": mismatch_open_closed,
        "mismatch_closed_open": mismatch_closed_open,
        "mean_dtw":             mean_dtw,
        "max_dtw":              max_dtw,
        "std_dtw":              std_dtw,
        "mean_corr":            mean_corr,
        "mean_combined":        mean_combined,
        "std_combined":         std_combined,
        "fake_ratio":           fake_ratio,
        "n_fake_segments":      n_fake_segments,
    }


# ── Full pipeline ────────────────────────────────────────────────
def run_pipeline(video_path, model_path, rf_model, scaler,
                 whisper_model, optimal_threshold=0.2027):
    """
    Full inference pipeline for one video.
    Returns prediction, confidence, fake segments, transcript, features.
    """
    # 1. Lip curve
    with load_landmarker(model_path) as landmarker:
        lip_curve, fps, frame_count = extract_lip_curve(
            video_path, landmarker
        )

    # 2. Audio + transcription
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    try:
        code = extract_audio(video_path, tmp_wav)
        if code != 0:
            raise RuntimeError("Audio extraction failed")
        result = transcribe_audio(tmp_wav, whisper_model)
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)

    # 3. Phoneme curve
    has_words = any(
        len(s.get("words", [])) > 0 for s in result["segments"]
    )
    phoneme_curve = (
        words_to_phoneme_curve(result, frame_count, fps)
        if has_words
        else np.zeros(frame_count, dtype=np.float32)
    )

    # 4. Features
    features = extract_features(
        lip_curve, phoneme_curve, fps, optimal_threshold
    )
    feature_order = [
        "pearson", "xcorr_peak", "lip_variance", "lip_mad",
        "lip_range", "phoneme_coverage", "peak_ratio", "mad_global",
        "mismatch_open_closed", "mismatch_closed_open", "mean_dtw",
        "max_dtw", "std_dtw", "mean_corr", "mean_combined",
        "std_combined", "fake_ratio", "n_fake_segments"
    ]
    X     = scaler.transform([[features[k] for k in feature_order]])
    proba = rf_model.predict_proba(X)[0][1]
    pred  = int(rf_model.predict(X)[0])

    # 5. Fake segments
    window_scores          = sliding_window_dtw_scores_v2(
        lip_curve, phoneme_curve, fps
    )
    fake_segments, fake_ratio = detect_fake_segments(
        window_scores, threshold=optimal_threshold
    )

    return {
        "prediction":    "fake" if pred == 1 else "real",
        "confidence":    round(float(proba), 4),
        "fake_ratio":    round(float(fake_ratio), 4),
        "fake_segments": [
            {
                "start": round(s, 2),
                "end":   round(e, 2),
                "score": round(sc, 4)
            }
            for s, e, sc in fake_segments
        ],
        "transcript": result["text"].strip(),
        "features":   {k: round(v, 4) for k, v in features.items()},
    }