"""
Modal GPU backend for the AI restore stage.

Render's cheap instances have no GPU, so the heavy GFPGAN restore is offloaded to
Modal (serverless GPU). This module is BOTH:
  * the deployable Modal app  ->  `modal deploy webapp/modal_restore.py`
  * the client the web app calls  ->  `restore_via_modal(src, dst, intensity)`

The web app sets RESTORE_BACKEND=modal and needs MODAL_TOKEN_ID / MODAL_TOKEN_SECRET
in its environment to invoke the deployed function.
"""
from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path

import modal

GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
MODEL_PATH = "/models/GFPGANv1.4.pth"

app = modal.App("douyin-restore")


def _bake_helpers():
    # Constructing GFPGANer once downloads the facexlib detection/parsing models,
    # baking them into the image so runtime has no cold-download.
    from gfpgan import GFPGANer
    GFPGANer(model_path=MODEL_PATH, upscale=2, arch="clean",
             channel_multiplier=2, bg_upsampler=None)


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "wget")
    # build tools + numpy first so basicsr's setup.py builds in a clean env
    .pip_install("setuptools<70", "wheel", "cython", "numpy==1.26.4")
    # torchvision 0.16.2 is the last release that ships functional_tensor.py as a
    # real module, so basicsr's imports work without any patching. Use the PyTorch
    # cu121 index so the wheels match Modal's T4 CUDA 12.1 environment.
    .pip_install(
        "torch==2.1.2", "torchvision==0.16.2",
        extra_options="--index-url https://download.pytorch.org/whl/cu121"
    )
    .pip_install("opencv-python-headless==4.9.0.80")
    # install the GFPGAN stack with build isolation OFF (uses the torch above)
    .pip_install("basicsr==1.4.2", "facexlib==0.3.0", "gfpgan==1.3.8",
                 "realesrgan==0.3.0", extra_options="--no-build-isolation")
    # GFPGAN deps may bump numpy; force it back to 1.x (torch 2.1.2 abi)
    .pip_install("numpy==1.26.4")
    .run_commands(f"mkdir -p /models && wget -q -O {MODEL_PATH} {GFPGAN_URL}")
    .run_function(_bake_helpers)
)


@app.function(image=image, gpu="T4", timeout=3600, retries=1)
def restore_bytes(video: bytes, intensity: float = 0.35) -> bytes:
    """Restore + 2× upscale every frame on the GPU; return the encoded video bytes
    (audio copied from the input). Frame loop mirrors src/restore.py."""
    import cv2
    import numpy as np
    from gfpgan import GFPGANer

    work = Path(tempfile.mkdtemp())
    src = work / "in.mp4"
    dst = work / "out.mp4"
    src.write_bytes(video)

    rest = GFPGANer(model_path=MODEL_PATH, upscale=2, arch="clean",
                    channel_multiplier=2, bg_upsampler=None)

    def enhance(bgr):
        _, _, out = rest.enhance(bgr, has_aligned=False, only_center_face=False,
                                 paste_back=True, weight=0.5)
        if intensity < 1.0:
            up = cv2.resize(bgr, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_CUBIC)
            out = cv2.addWeighted(out, intensity, up, 1.0 - intensity, 0.0)
        return out

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("no frames in input")
    first = enhance(frame)
    h, w = first.shape[:2]

    ff = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
          "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
          "-i", str(src), "-map", "0:v", "-map", "1:a?", "-c:a", "copy", "-shortest",
          "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", str(dst)]
    proc = subprocess.Popen(ff, stdin=subprocess.PIPE)
    proc.stdin.write(np.ascontiguousarray(first, dtype=np.uint8).tobytes())
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        proc.stdin.write(np.ascontiguousarray(enhance(frame), dtype=np.uint8).tobytes())
    cap.release()
    proc.stdin.close(); proc.wait()
    return dst.read_bytes()


# --- client side (runs inside the Render web app) -------------------------------
def restore_via_modal(src: str, dst: str, intensity: float) -> None:
    """Send the local clip to the deployed Modal GPU function and write the result."""
    fn = modal.Function.from_name("douyin-restore", "restore_bytes")
    out = fn.remote(Path(src).read_bytes(), intensity)
    Path(dst).write_bytes(out)


@app.local_entrypoint()
def smoke(path: str):
    """Quick test: `modal run webapp/modal_restore.py --path some.mp4`."""
    out = restore_bytes.remote(Path(path).read_bytes(), 0.35)
    Path("modal_restore_out.mp4").write_bytes(out)
    print(f"wrote modal_restore_out.mp4 ({len(out)} bytes)")
