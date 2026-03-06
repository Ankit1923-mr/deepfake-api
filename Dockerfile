FROM python:3.11-slim

# 1. Install system dependencies and clean up in one step to keep the image lean
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    espeak-ng \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


# 2. Upgrade pip and install the pre-compiled Whisper wheel directly from PyPI
# 2. Install a compatible setuptools and force pip to use it via --no-build-isolation
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "setuptools<70.0.0" wheel && \
    pip install --no-cache-dir --no-build-isolation openai-whisper==20231117

# 3. Copy requirements first to leverage Docker's layer caching
COPY requirements.txt .

# 4. Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application files
# (Doing this AFTER requirements ensures code changes don't trigger a full pip reinstall)
COPY main.py pipeline.py rf_model.pkl scaler.pkl face_landmarker.task ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]