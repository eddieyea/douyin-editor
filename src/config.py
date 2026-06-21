"""
Configuration for the Douyin/Xiaohongshu video editor.

Two layers:
  1. PARAMS  – the 剪映 (CapCut/Jianying) slider values the human operator uses.
               These are on 剪映's own scale (mostly -100..+100, sharpen 0..100).
  2. MAP     – coefficients translating a 剪映 slider unit into the matching
               ffmpeg filter value. These are *calibrated* against raw+final
               clip pairs (see src/calibrate.py); the defaults below are
               documented first-guesses and get overwritten by calibration.

Anything here can be overridden by a JSON file passed to the CLI (--config),
so the operator never edits Python.
"""
from __future__ import annotations
import copy
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


# ---------------------------------------------------------------------------
# 剪映 slider values from the human operator's spec.
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    # 亮度 = "根据视频情况调节" -> handled by auto-brightness, see grade.py.
    #   manual_brightness is an *extra* offset on 剪映's -100..100 scale, added
    #   on top of the auto correction. Leave 0 to let auto do all the work.
    "brightness_auto": True,
    "brightness_target": 0.50,   # target mean luma (0..1) for auto mode
    "manual_brightness": 0,      # 剪映 -100..100, extra trim
    "sharpen": 20,               # 锐化 +20
    "contrast": 13,              # 对比度 +13
    "saturation": 8,             # 饱和度 +5~10 -> default mid 8
    "temperature": -9,           # 色温 -9 (negative = cooler/bluer)

    # 美颜 (only applied when --beauty is on or needs_beauty in a job file)
    "beauty_skin": 20,           # 美颜 20%
    "beauty_slim": 10,           # 瘦脸 10%
}


# ---------------------------------------------------------------------------
# Slider-unit -> ffmpeg-value mapping coefficients (calibratable).
# ---------------------------------------------------------------------------
DEFAULT_MAP = {
    # eq contrast = 1 + (contrast/100) * k_contrast
    "k_contrast": 0.50,
    # eq saturation = 1 + (saturation/100) * k_saturation
    "k_saturation": 1.00,
    # eq brightness add = (manual_brightness/100) * k_brightness
    "k_brightness": 0.15,
    # unsharp luma_amount = (sharpen/100) * k_sharpen
    "k_sharpen": 1.50,
    "unsharp_msize": 5,          # luma_msize_x / luma_msize_y
    # colortemperature Kelvin = 6500 - temperature * k_temp_kelvin
    #   temperature is negative for "cooler", which pushes Kelvin HIGHER,
    #   and the colortemperature filter renders higher Kelvin as a cooler
    #   (bluer) tint -> sign is correct.
    "k_temp_kelvin": 50.0,
    "temp_mix": 1.0,
    "temp_preserve_lightness": 1.0,
}


# ---------------------------------------------------------------------------
# Output / export settings.
# ---------------------------------------------------------------------------
DEFAULT_EXPORT = {
    "width": 1080,
    "height": 1920,
    "fps": 30,
    "fit": "cover",              # "cover" (fill+crop) or "contain" (fit+pad)
    "hdr": False,                # True -> HDR10 path (src/hdr.py)
    "video_codec": "libx264",    # SDR default; HDR path overrides to libx265
    "crf": 18,
    "preset": "slow",
    "audio_codec": "aac",
    "audio_bitrate": "192k",
    "pad_color": "black",
}


# Auto-trim (dead air / non-performance removal).
DEFAULT_TRIM = {
    "cut_silence_ms": 600,   # drop non-speech gaps longer than this
    "pad_ms": 200,           # keep this much padding around each speech run
    "threshold": 0.5,        # Silero VAD speech probability threshold
    "end_pad_ms": 1000,      # extend the final clip end ~1s past the last word
}


@dataclass
class Settings:
    params: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_PARAMS))
    map: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_MAP))
    export: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_EXPORT))
    trim: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_TRIM))

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Settings":
        s = cls()
        if path:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            for key in ("params", "map", "export", "trim"):
                if key in data:
                    getattr(s, key).update(data[key])
        return s

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# Word swaps for subtitle censorship-evasion (platform moderation dodging).
WORD_SWAPS = {
    "嫖娼": "PC",
    "暴力": "B力",
    "修复": "X复",
    "出轨": "出🚗",
}
