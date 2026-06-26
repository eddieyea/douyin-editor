FROM python:3.10-slim

# system deps: ffmpeg for video processing, libgl1+libglib2.0-0 for OpenCV/mediapipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# source + static assets (fonts, face_landmarker model)
COPY src/ src/
COPY webapp/ webapp/
COPY assets/ assets/

# jobs dir must exist at startup; Render ephemeral FS is fine for single-session use
RUN mkdir -p webapp/jobs

# Pre-bake faster-whisper "base" model so the first job doesn't block on HuggingFace
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8'); print('Whisper base model cached.')"

ENV PYTHONUNBUFFERED=1

# port 8000 is Render's expected default for Docker web services
CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]
