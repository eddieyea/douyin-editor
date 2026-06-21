"""
Auto-trim: cut dead air / non-performance from a talking-head clip.

A scripted talking-head video is "performing" when the actor is speaking. The
hiccups and dead spots (long pauses, fiddling with the phone, the camera turn-off
at the end) are non-speech. We detect speech with Silero VAD (bundled in
faster-whisper), keep the speech regions (padded), and drop the long silences.

This is a speech-based proxy: it removes silence/dead air, not on-camera flubs
where the actor is still talking. Thresholds are tunable in config.py.
"""
from __future__ import annotations
import subprocess
from pathlib import Path

from .grade import FFMPEG


def speech_keep_intervals(src: str, cut_silence_ms: int = 600, pad_ms: int = 200,
                          threshold: float = 0.5, end_pad_ms: int = 1000
                          ) -> tuple[list, float]:
    """Return (keep_intervals_seconds, total_duration). Silences longer than
    cut_silence_ms are dropped; each speech run is padded by pad_ms. The final
    kept interval is extended by end_pad_ms so the last word finishes naturally
    (VAD tends to clip the trailing-off of the last word) and the video ends ~1s
    after it rather than cutting abruptly."""
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import get_speech_timestamps, VadOptions
    audio = decode_audio(src, sampling_rate=16000)
    duration = len(audio) / 16000.0
    opts = VadOptions(threshold=threshold,
                      min_silence_duration_ms=cut_silence_ms,
                      speech_pad_ms=pad_ms)
    ts = get_speech_timestamps(audio, opts, 16000)
    intervals = [(t["start"] / 16000.0, t["end"] / 16000.0) for t in ts]
    if intervals:
        s, e = intervals[-1]
        intervals[-1] = (s, min(duration, e + end_pad_ms / 1000.0))
    return intervals, duration


def _select_expr(intervals: list) -> str:
    return "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in intervals)


def trim_video(src: str, dst: str, intervals: list, crf: int = 16) -> str:
    """Concatenate the keep intervals into dst (re-timed, A/V kept in sync)."""
    if not intervals:
        raise ValueError("no speech intervals found to keep")
    expr = _select_expr(intervals)
    cmd = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
        "-vf", f"select='{expr}',setpts=N/FRAME_RATE/TB",
        "-af", f"aselect='{expr}',asetpts=N/SR/TB",
        "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", dst,
    ]
    subprocess.run(cmd, check=True)
    kept = sum(e - s for s, e in intervals)
    return f"kept {kept:.1f}s across {len(intervals)} segments -> {dst}"
