"""
AI face restoration / super-resolution (GFPGAN), for low-res sources.

The studio clips arrive WeChat-compressed (544×960), so a 2× upscale to 1080p is
soft. GFPGAN restores facial detail and (optionally, via RealESRGAN) upscales the
background, adding real texture back to faces/hair instead of just interpolating.

Runs on Apple-Silicon GPU via MPS (PYTORCH_ENABLE_MPS_FALLBACK covers any op the
Metal backend lacks). It's still heavy per-frame, so this is an opt-in stage.
"""
from __future__ import annotations
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from .grade import FFMPEG

ROOT = Path(__file__).resolve().parent.parent
GFPGAN_MODEL = str(ROOT / "assets" / "gfpgan" / "GFPGANv1.4.pth")
REALESRGAN_MODEL = str(ROOT / "assets" / "gfpgan" / "RealESRGAN_x2plus.pth")


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Restorer:
    def __init__(self, upscale: int = 2, bg: bool = True, device: str | None = None):
        from gfpgan import GFPGANer
        dev = torch.device(device or _device())
        bg_upsampler = None
        if bg and Path(REALESRGAN_MODEL).exists():
            try:
                from realesrgan import RealESRGANer
                from basicsr.archs.rrdbnet_arch import RRDBNet
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                num_block=23, num_grow_ch=32, scale=2)
                bg_upsampler = RealESRGANer(
                    scale=2, model_path=REALESRGAN_MODEL, model=model,
                    tile=400, tile_pad=10, pre_pad=0, half=False, device=dev)
            except Exception:
                bg_upsampler = None
        try:
            self.r = GFPGANer(model_path=GFPGAN_MODEL, upscale=upscale, arch="clean",
                              channel_multiplier=2, bg_upsampler=bg_upsampler, device=dev)
        except TypeError:  # older signature without device kw
            self.r = GFPGANer(model_path=GFPGAN_MODEL, upscale=upscale, arch="clean",
                              channel_multiplier=2, bg_upsampler=bg_upsampler)

    def enhance(self, bgr: np.ndarray, weight: float = 0.5,
                intensity: float = 1.0) -> np.ndarray:
        """Restore + 2× upscale, then blend back toward the plain upscaled original
        so the effect can be dialed down (intensity 1.0 = full GFPGAN, lower = more
        natural). `weight` is GFPGAN's own fidelity (0=max restoration, 1=input)."""
        _, _, out = self.r.enhance(bgr, has_aligned=False, only_center_face=False,
                                   paste_back=True, weight=weight)
        if intensity < 1.0:
            up = cv2.resize(bgr, (out.shape[1], out.shape[0]),
                            interpolation=cv2.INTER_CUBIC)
            out = cv2.addWeighted(out, intensity, up, 1.0 - intensity, 0.0)
        return out


def process_video(src: str, dst: str, weight: float = 0.5, intensity: float = 0.35,
                  bg: bool = True, audio_src: str | None = None, crf: int = 14) -> str:
    """Restore + 2× upscale every frame; pipe to ffmpeg at the restored native
    size (later stages crop/scale to 1080×1920). Audio copied from audio_src."""
    rest = Restorer(upscale=2, bg=bg)
    cap = cv2.VideoCapture(src)
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("no frames")
    first = rest.enhance(frame, weight=weight, intensity=intensity)
    h, w = first.shape[:2]

    ff = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
          "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
          "-r", f"{in_fps}", "-i", "-"]
    if audio_src:
        ff += ["-i", audio_src, "-map", "0:v", "-map", "1:a?", "-c:a", "copy", "-shortest"]
    ff += ["-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
           "-pix_fmt", "yuv420p", dst]
    proc = subprocess.Popen(ff, stdin=subprocess.PIPE)

    def write(img):
        proc.stdin.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())

    write(first)
    n = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        write(rest.enhance(frame, weight=weight, intensity=intensity))
        n += 1
    cap.release()
    proc.stdin.close(); proc.wait()
    return f"{n} frames restored ({w}x{h}) -> {dst}"
