# Douyin / Xiaohongshu video editor

Automates the manual 剪映 (CapCut) finishing workflow for 抖音/小红书 short videos:
color grade, censorship-evasion subtitle swaps, beauty, and 1080p30 export.

## Setup
Self-contained in `venv/` (Python 3.9). ffmpeg is the bundled `imageio-ffmpeg`
binary, symlinked at `bin/ffmpeg`.

```bash
./venv/bin/python -m src.cli grade <input> [options]
```

## What works now (all four features)
- **Color grade** (`src/grade.py`): 剪映 sliders → ffmpeg `eq`+`colortemperature`+`unsharp`.
  Brightness is auto (measures source luma, "根据视频情况调节"). Defaults:
  锐化+20, 对比度+13, 饱和度+8, 色温-9.
- **Export**: 1080×1920, 30fps, `cover` (fill+crop) or `--contain`.
  SDR = H.264/AAC; `--hdr` = **智能HDR** 10-bit HEVC HDR10 (BT.2020/PQ) — see `src/hdr.py`.
- **Subtitles** (`src/subtitles.py` + `src/captions.py`): faster-whisper Chinese
  ASR → word swaps → Douyin-style color captions burned in.
  - Swaps: 嫖娼→PC, 暴力→B力, 修复→X复, 出轨→出🚗
  - Captions are rendered as PNG overlays (Pillow) so 出🚗 shows the **color**
    car emoji — libass can't rasterize Apple color emoji.
- **美颜/瘦脸** (`src/beauty.py`): MediaPipe FaceLandmarker (model in `assets/`)
  → skin smoothing (bilateral, masked to skin) + face-slim warp. Pre-pass before grade.

- **Auto-trim** (`src/trim.py`): Silero VAD (via faster-whisper) removes dead air /
  non-performance (long pauses, end-of-clip phone fiddling). Runs first, before beauty.
  Tunable in `config.py` (`cut_silence_ms`, `pad_ms`).

Caption layout & timing:
- Word-accurate timing from Whisper word-timestamps (not drifting segment times).
- Chinese word segmentation via **jieba** (+ trailing-particle merge) so a caption
  never breaks inside a word (盯着, 手机 stay whole).
- Breaks at sentence punctuation and at speech pauses (>0.30s), so a new sentence
  starts a new caption (e.g. 是 after 什么？).
- One line per caption: a small overflow (≤4 words) shrinks the font slightly;
  a big overflow splits into balanced sequential lines (no orphan fragments).
- Short captions are held ≥0.5s on screen (no flicker).
- Punctuation stripped except ? / ？.

```bash
# grade + export only
./venv/bin/python -m src.cli grade clip.mov -o out.mp4
# everything: trim + 美颜/瘦脸 + auto captions + grade + 智能HDR
./venv/bin/python -m src.cli grade clip.mov --trim --beauty --subtitle --hdr --model small
```

- **AI restore** (`src/restore.py`): GFPGAN face restoration + 2× upscale on the
  Metal GPU (MPS), for low-res/WeChat-compressed sources. ~0.57s/frame (~47 min
  for a full clip). `--restore-weight` 0..1 (0=max detail, 1=closest to input).
  Auto-disables 美颜 skin-smoothing (keeps 瘦脸) so it doesn't smear recovered detail.

Pipeline order: `--trim` → `--restore` → `--beauty` → grade → `--subtitle` captions → export.

Flags: `--trim` `--beauty` `--subtitle` `--hdr` `--contain` `--no-auto-bright`
`--model {tiny,base,small,medium}` `--config cfg.json` `--dry-run`

## Tuning
All 剪映 slider values and the slider→ffmpeg coefficients live in `src/config.py`
and can be overridden with `--config cfg.json`. The coefficients are calibrated
against raw+final clip pairs (see TODO: `src/calibrate.py`).

## Notes / next
- Grade coefficients use conventional 剪映→ffmpeg mappings. Precise calibration
  needs a 剪映 **export master** (not the platform-recompressed copy) — see
  `calibration/FINDINGS.md`.
- HDR is a correct SDR→HDR10 wrap (SDR source has no real HDR detail to recover).
- All slider values + mapping coefficients live in `src/config.py` (override via `--config`).
