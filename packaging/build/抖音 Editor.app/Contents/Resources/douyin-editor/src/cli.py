"""
CLI entry point.

    python -m src.cli grade  <input> [-o output.mp4] [--config cfg.json]
                             [--contain] [--no-auto-bright] [--dry-run]

More subcommands (subtitles, beauty, all) land with their modules.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from .config import Settings
from .pipeline import run_job, JobOptions


def _default_out(src: str) -> str:
    p = Path(src)
    outdir = Path(__file__).resolve().parent.parent / "output"
    outdir.mkdir(exist_ok=True)
    return str(outdir / f"{p.stem}_edited.mp4")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="douyin-editor")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("grade", help="color grade + 1080p30 export")
    g.add_argument("input")
    g.add_argument("-o", "--output")
    g.add_argument("--config", help="JSON overrides for params/map/export")
    g.add_argument("--contain", action="store_true",
                   help="fit+pad instead of fill+crop")
    g.add_argument("--no-auto-bright", action="store_true")
    g.add_argument("--subtitle", action="store_true",
                   help="auto ASR -> word swaps -> burn in color captions")
    g.add_argument("--trim", action="store_true",
                   help="auto-cut dead air / non-performance (VAD-based)")
    g.add_argument("--restore", action="store_true",
                   help="AI face restoration + 2x upscale (GFPGAN) for low-res sources")
    g.add_argument("--restore-intensity", type=float, default=0.35,
                   help="how much AI restoration to blend in, 0..1 "
                        "(0=natural/soft, ~0.35=subtle clarity, 1=full/porcelain)")
    g.add_argument("--beauty", action="store_true",
                   help="美颜/瘦脸 pre-pass (uses config beauty_skin/beauty_slim)")
    g.add_argument("--hdr", action="store_true",
                   help="智能HDR: export 10-bit HEVC HDR10 (BT.2020 PQ)")
    g.add_argument("--model", default="small",
                   help="faster-whisper model size (tiny/base/small/medium)")

    args = ap.parse_args(argv)

    if args.cmd == "grade":
        settings = Settings.load(args.config)
        opts = JobOptions(
            trim=args.trim, restore=args.restore,
            restore_intensity=args.restore_intensity, beauty=args.beauty,
            subtitle=args.subtitle, hdr=args.hdr, contain=args.contain,
            auto_bright=not args.no_auto_bright, model=args.model,
        )
        out = args.output or _default_out(args.input)
        run_job(args.input, out, opts, settings=settings,
                progress=lambda m: print(f"· {m}"))
        print(f"✓ wrote {out}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
