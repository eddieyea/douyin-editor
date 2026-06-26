"""
Pipeline orchestrator: raw clip -> (beauty) -> grade -> captions -> export.

Stage order matches the human 剪映 workflow:
    [beauty]    geometry + skin on raw frames        (task #4, hook below)
    [grade]     color / sharpen / temperature        (src/grade.py)
    [captions]  ASR -> word swaps -> color overlays   (src/subtitles, src/captions)
    [export]    1080x1920 30fps  (SDR now; HDR = task #5)

Captions are full-frame transparent PNGs overlaid with per-segment timing, so
出🚗 renders in color (libass can't rasterize Apple's color emoji).
"""
from __future__ import annotations
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .grade import FFMPEG, build_grade_filters, probe_mean_luma
from .captions import CaptionStyle, render_items, build_caption_items
from .subtitles import transcribe_words, to_word_units, apply_swaps, strip_punct
from .hdr import HDR_CONVERT, hdr_encode_args


def _encode_args(export: dict) -> list[str]:
    if export["hdr"]:
        return hdr_encode_args(export)
    return [
        "-c:v", export["video_codec"],
        "-pix_fmt", "yuv420p",
        "-crf", str(export["crf"]),
        "-preset", export["preset"],
        "-r", str(export["fps"]),
        "-c:a", export["audio_codec"],
        "-b:a", export["audio_bitrate"],
        "-movflags", "+faststart",
    ]


def build_captions(src: str, settings: Settings, model_size: str = "small",
                   out_dir: str | None = None) -> list[dict]:
    """ASR -> word swaps -> rendered caption PNGs. Returns overlay items."""
    exp = settings.export
    style = CaptionStyle(exp["width"], exp["height"])
    # word-level timestamps -> accurate timing; build one-line items (shrink-to-fit
    # short tails) with punctuation stripped (keep ?/？) and word swaps applied
    segments = transcribe_words(src, model_size=model_size)
    word_units = to_word_units(segments)  # jieba words -> never break inside a word
    transform = lambda t: apply_swaps(strip_punct(t))
    items = build_caption_items(word_units, style, transform=transform)
    out_dir = out_dir or str(Path(FFMPEG).parent.parent / "tmp" / "captions")
    return render_items(items, style, out_dir)


def process(src: str, dst: str, settings: Settings,
            captions: list[dict] | None = None, dry_run: bool = False) -> str:
    params, mp, export = settings.params, settings.map, settings.export

    mean_luma = None
    if params.get("brightness_auto", True):
        mean_luma = probe_mean_luma(src)
    grade = build_grade_filters(params, mp, export, src_mean_luma=mean_luma)

    cmd = [FFMPEG, "-y", "-i", src]
    for item in (captions or []):
        cmd += ["-i", item["png"]]

    if captions:
        # filter_complex: grade base, then chain timed overlays
        parts = [f"[0:v]{','.join(grade)}[base]"]
        prev = "base"
        for i, item in enumerate(captions, start=1):
            label = f"v{i}"
            en = f"between(t\\,{item['start']:.3f}\\,{item['end']:.3f})"
            parts.append(f"[{prev}][{i}:v]overlay=x=0:y=0:enable='{en}'[{label}]")
            prev = label
        if export["hdr"]:
            parts.append(f"[{prev}]{HDR_CONVERT}[vout]")
            prev = "vout"
        filtergraph = ";".join(parts)
        cmd += ["-filter_complex", filtergraph, "-map", f"[{prev}]", "-map", "0:a?"]
    else:
        vf = list(grade)
        if export["hdr"]:
            vf.append(HDR_CONVERT)
        cmd += ["-vf", ",".join(vf)]

    cmd += [*_encode_args(export), dst]

    printable = " ".join(cmd)
    if dry_run:
        return printable
    subprocess.run(cmd, check=True)
    return printable


# ---------------------------------------------------------------------------
# High-level orchestration shared by the CLI and the web app.
# ---------------------------------------------------------------------------
@dataclass
class JobOptions:
    trim: bool = False
    restore: bool = False
    restore_intensity: float = 0.35
    beauty: bool = False
    subtitle: bool = True
    hdr: bool = False
    contain: bool = False
    auto_bright: bool = True
    model: str = "small"


def _local_restore(src: str, dst: str, intensity: float) -> None:
    from .restore import process_video
    process_video(src, dst, intensity=intensity, bg=False, audio_src=src)


def run_job(src: str, dst: str, opts: JobOptions, settings: Settings | None = None,
            tmpdir: str | None = None, progress=lambda m: None,
            restore_backend=None) -> str:
    """Full pipeline: trim -> restore -> beauty -> grade -> captions -> export.

    `progress(msg)` is called at each stage. `restore_backend(src,dst,intensity)`
    defaults to local GFPGAN; the web app passes a Modal/GPU-backed one.
    """
    settings = settings or Settings()
    if opts.contain:
        settings.export["fit"] = "contain"
    if not opts.auto_bright:
        settings.params["brightness_auto"] = False
    if opts.hdr:
        settings.export["hdr"] = True

    tmp = Path(tmpdir or (Path(FFMPEG).parent.parent / "tmp"))
    tmp.mkdir(parents=True, exist_ok=True)
    src_in = src

    if opts.trim:
        from .trim import speech_keep_intervals, trim_video
        progress("Trimming dead air…")
        iv, dur = speech_keep_intervals(src_in, **settings.trim)
        out = str(tmp / "_trimmed.mp4")
        trim_video(src_in, out, iv)
        src_in = out

    if opts.restore:
        progress("AI restore + upscale…")
        out = str(tmp / "_restored.mp4")
        (restore_backend or _local_restore)(src_in, out, opts.restore_intensity)
        src_in = out
        if opts.beauty and settings.params["beauty_skin"] > 0:
            settings.params["beauty_skin"] = 0  # restore already cleans skin

    if opts.beauty:
        from .beauty import process_video as beauty_pass
        progress("美颜/瘦脸…")
        out = str(tmp / "_beauty.mp4")
        beauty_pass(src_in, out, skin=settings.params["beauty_skin"],
                    slim=settings.params["beauty_slim"], audio_src=src_in)
        src_in = out

    captions = None
    if opts.subtitle:
        progress("Transcribing + captions…")
        captions = build_captions(src_in, settings, model_size=opts.model,
                                  out_dir=str(tmp / "captions"))

    progress("Encoding final…")
    process(src_in, dst, settings, captions=captions)
    progress("Done")
    return dst
