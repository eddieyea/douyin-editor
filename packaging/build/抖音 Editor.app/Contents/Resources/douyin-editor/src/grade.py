"""
Color grading + geometry: translate 剪映 sliders into an ffmpeg filtergraph.

Pipeline order (matches what the 剪映 operator does):
    scale/crop to 1080x1920  ->  eq (brightness/contrast/saturation)
    ->  colortemperature (色温)  ->  unsharp (锐化)

Auto-brightness: 剪映 brightness is "根据视频情况调节", so we measure the source
mean luma with ffmpeg's signalstats and nudge it toward a target, instead of a
fixed slider value.
"""
from __future__ import annotations
import re
import subprocess
from pathlib import Path

def _resolve_ffmpeg() -> str:
    # Local macOS dev build first; fall back to imageio-ffmpeg (cross-platform
    # PyPI binary) then system ffmpeg (available in the Render/Linux container).
    local = Path(__file__).resolve().parent.parent / "bin" / "ffmpeg"
    if local.exists() and local.stat().st_size > 0:
        import os, stat
        if local.stat().st_mode & stat.S_IXUSR:
            return str(local)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    import shutil
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    raise FileNotFoundError("ffmpeg not found — install imageio-ffmpeg or apt install ffmpeg")

FFMPEG = _resolve_ffmpeg()


def probe_mean_luma(src: str, sample_frames: int = 200) -> float:
    """Return mean luma in 0..1 by averaging signalstats YAVG over sampled frames."""
    # Sample evenly across the clip to stay fast on long videos.
    cmd = [
        FFMPEG, "-hide_banner", "-nostats", "-i", src,
        "-vf", f"signalstats,metadata=print:key=lavfi.signalstats.YAVG",
        "-frames:v", str(sample_frames), "-an", "-f", "null", "-",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True).stderr
    vals = [float(m) for m in re.findall(r"YAVG=([0-9.]+)", out)]
    if not vals:
        return 0.5
    avg = sum(vals) / len(vals)
    return avg / 255.0  # YAVG is in 0..255 for 8-bit


def _scale_filter(export: dict) -> str:
    w, h = export["width"], export["height"]
    if export["fit"] == "contain":
        return (
            f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color={export['pad_color']},"
            f"setsar=1"
        )
    # cover: fill the frame, center-crop the overflow
    return (
        f"scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},setsar=1"
    )


def build_grade_filters(params: dict, mp: dict, export: dict,
                        src_mean_luma: float | None = None) -> list[str]:
    """Return an ordered list of filter strings (joined with ',' by caller)."""
    filters = [_scale_filter(export)]

    # --- brightness (additive eq term) ---
    bright = 0.0
    if params.get("brightness_auto", True) and src_mean_luma is not None:
        # Move mean luma toward target; eq brightness is an additive 0..1 shift.
        bright += params["brightness_target"] - src_mean_luma
    bright += (params.get("manual_brightness", 0) / 100.0) * mp["k_brightness"]
    bright = max(-0.5, min(0.5, bright))  # clamp to sane range

    contrast = 1.0 + (params["contrast"] / 100.0) * mp["k_contrast"]
    saturation = 1.0 + (params["saturation"] / 100.0) * mp["k_saturation"]

    filters.append(
        f"eq=brightness={bright:.4f}:contrast={contrast:.4f}:saturation={saturation:.4f}"
    )

    # --- 色温 color temperature ---
    if params.get("temperature", 0):
        kelvin = 6500.0 - params["temperature"] * mp["k_temp_kelvin"]
        kelvin = max(1000.0, min(40000.0, kelvin))
        filters.append(
            f"colortemperature=temperature={kelvin:.0f}"
            f":mix={mp['temp_mix']}:pl={mp['temp_preserve_lightness']}"
        )

    # --- 锐化 sharpen ---
    if params.get("sharpen", 0) > 0:
        amount = (params["sharpen"] / 100.0) * mp["k_sharpen"]
        m = mp["unsharp_msize"]
        filters.append(f"unsharp=lx={m}:ly={m}:la={amount:.3f}")

    return filters
