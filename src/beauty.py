"""
美颜 / 瘦脸 — skin smoothing + face slimming via MediaPipe FaceLandmarker.

This mediapipe build (0.10.35) only ships the new Tasks API, so we use
FaceLandmarker (478-point topology, same indices as the old FaceMesh) with the
downloaded model at assets/face_landmarker.task.

  smooth_skin()  edge-preserving (bilateral) blur, masked to skin only
                 (face oval minus eyes/brows/lips), blended at 美颜 strength.
  slim_face()    local-translation warp pulling the cheek/jaw contour inward
                 toward the face centerline at 瘦脸 strength.

process_video() runs per frame and pipes the result to the bundled ffmpeg as a
near-lossless intermediate (audio copied from the original); the colour grade +
captions then run on top in the normal pipeline.
"""
from __future__ import annotations
import subprocess
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .grade import FFMPEG

MODEL_PATH = str(Path(__file__).resolve().parent.parent / "assets" / "face_landmarker.task")

# --- landmark index sets (MediaPipe 468/478 topology) ---
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
             365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
             132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386,
            385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159,
             158, 157, 173]
LEFT_BROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
RIGHT_BROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402,
        317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415,
        310, 311, 312, 13, 82, 81, 80, 191, 78]

# widest cheek / jaw landmarks per side
LEFT_JAW = [234, 93, 132, 58, 172]
RIGHT_JAW = [454, 323, 361, 288, 397]
CENTER_REF = 168  # nose bridge -> face vertical centerline

SKIN_MAX = 0.9          # blend weight at 100%
SLIM_MAX_FRAC = 0.06    # cheek pulled this fraction of face width at 100%
SLIM_RADIUS_FRAC = 0.22 # warp radius as fraction of face width


def _pts(landmarks, idxs, w, h):
    return np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in idxs],
                    dtype=np.float32)


def _all_pts(landmarks, w, h):
    return np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)


def _skin_mask(landmarks, w, h) -> np.ndarray:
    hull = cv2.convexHull(_pts(landmarks, FACE_OVAL, w, h).astype(np.int32))
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    for idx in (LEFT_EYE, RIGHT_EYE, LEFT_BROW, RIGHT_BROW, LIPS):
        h2 = cv2.convexHull(_pts(landmarks, idx, w, h).astype(np.int32))
        cv2.fillConvexPoly(mask, h2, 0)
    # feather only (no dilate — dilation would regrow skin over the carved eyes)
    mask = cv2.GaussianBlur(mask, (0, 0), w * 0.01)
    return mask


def smooth_skin(frame, landmarks, strength_pct: float) -> np.ndarray:
    if strength_pct <= 0:
        return frame
    h, w = frame.shape[:2]
    mask = _skin_mask(landmarks, w, h).astype(np.float32) / 255.0
    d = max(5, int(w * 0.012))
    smoothed = cv2.bilateralFilter(frame, d=d, sigmaColor=45, sigmaSpace=d)
    a = (strength_pct / 100.0) * SKIN_MAX
    m = (mask * a)[..., None]
    return (frame.astype(np.float32) * (1 - m) + smoothed.astype(np.float32) * m).astype(np.uint8)


def slim_face(frame, landmarks, strength_pct: float) -> np.ndarray:
    if strength_pct <= 0:
        return frame
    h, w = frame.shape[:2]
    P = _all_pts(landmarks, w, h)
    face_w = float(np.linalg.norm(P[454] - P[234]))
    cx = P[CENTER_REF][0]
    frac = (strength_pct / 100.0) * SLIM_MAX_FRAC
    radius = face_w * SLIM_RADIUS_FRAC

    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    map_x, map_y = gx.copy(), gy.copy()

    for idxs in (LEFT_JAW, RIGHT_JAW):
        for i in idxs:
            c = P[i]
            m = np.array([c[0] + (cx - c[0]) * frac, c[1]], dtype=np.float32)
            dx, dy = m[0] - c[0], m[1] - c[1]
            distx, disty = gx - c[0], gy - c[1]
            d2 = distx * distx + disty * disty
            r2 = radius * radius
            within = d2 < r2
            denom = (r2 - d2) + (dx * dx + dy * dy)
            ratio = np.where(within, ((r2 - d2) / np.maximum(denom, 1e-6)) ** 2, 0.0)
            map_x -= (ratio * dx).astype(np.float32)
            map_y -= (ratio * dy).astype(np.float32)

    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def make_landmarker(running_mode):
    base = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base, running_mode=running_mode, num_faces=1)
    return mp_vision.FaceLandmarker.create_from_options(opts)


def detect_image(frame_bgr):
    """Single-image detection -> landmark list (or None). frame is BGR."""
    lm = make_landmarker(mp_vision.RunningMode.IMAGE)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = lm.detect(img)
    lm.close()
    return res.face_landmarks[0] if res.face_landmarks else None


def process_video(src: str, dst: str, skin: float = 20, slim: float = 10,
                  audio_src: str | None = None, crf: int = 14) -> str:
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ff = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
          "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
          "-r", f"{fps}", "-i", "-"]
    if audio_src:
        ff += ["-i", audio_src, "-map", "0:v", "-map", "1:a?", "-c:a", "copy", "-shortest"]
    ff += ["-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
           "-pix_fmt", "yuv420p", dst]
    proc = subprocess.Popen(ff, stdin=subprocess.PIPE)

    lm = make_landmarker(mp_vision.RunningMode.VIDEO)
    n = 0
    last_ts = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = max(last_ts + 1, int(n / fps * 1000))
        last_ts = ts
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = lm.detect_for_video(img, ts)
        if res.face_landmarks:
            pts = res.face_landmarks[0]
            frame = slim_face(frame, pts, slim)
            frame = smooth_skin(frame, pts, skin)
        proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        n += 1
    cap.release(); lm.close()
    proc.stdin.close(); proc.wait()
    return f"{n} frames -> {dst}"
