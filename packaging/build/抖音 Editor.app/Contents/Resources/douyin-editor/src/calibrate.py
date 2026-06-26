"""
Calibrate slider->ffmpeg coefficients from a raw + 剪映-export-master pair.

Measures the colour transform raw->final and solves the eq/colortemperature
coefficients so that applying the operator's sliders to a raw clip reproduces it.

Best signal: a 剪映 EXPORT MASTER (the file 剪映 exports, before uploading to the
platform), ideally **grade-only — beauty OFF, no captions** — so the measured
transform is purely the colour grade. A platform-republished copy is too lossy
(re-encode + downscale) and beauty smoothing cancels the contrast boost.

Frames are extracted once and measured with OpenCV; the colour-temperature match
is a ternary search over a small frame subset, so the whole run is a few seconds.

    ./venv/bin/python -m src.calibrate RAW.mp4 FINAL.mp4 [--apply]
    --apply  writes calibration/calibrated.json (use it via: --config calibration/calibrated.json)
"""
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .grade import FFMPEG
from .config import DEFAULT_PARAMS, DEFAULT_MAP

ROOT = Path(__file__).resolve().parent.parent
CROP = "crop=iw:ih*0.60:0:0"   # top 60% -> excludes caption band
COMMON_W = 480                  # normalise both clips to the same width


def _extract(src: str, out_dir: Path, n: int = 24) -> list[Path]:
    # n evenly spaced frames; thumbnail to a common width so stats are comparable
    vf = f"{CROP},scale={COMMON_W}:-2"
    subprocess.run(
        [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
         "-vf", f"thumbnail=n=300,{vf}", "-frames:v", str(n),
         "-vsync", "vfr", str(out_dir / "f_%03d.png")],
        check=True,
    )
    fs = sorted(out_dir.glob("f_*.png"))
    if not fs:  # fallback: simple periodic sampling
        subprocess.run(
            [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
             "-vf", f"fps=1/5,{vf}", str(out_dir / "g_%03d.png")], check=True)
        fs = sorted(out_dir.glob("g_*.png"))
    return fs


def _stats_of(frames: list[Path]) -> dict:
    ys, ystd, ss, rs, gs, bs = [], [], [], [], [], []
    for fp in frames:
        bgr = cv2.imread(str(fp))
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ys.append(float(gray.mean())); ystd.append(float(gray.std()))
        ss.append(float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1].mean()))
        b, g, r = bgr[..., 0].mean(), bgr[..., 1].mean(), bgr[..., 2].mean()
        rs.append(float(r)); gs.append(float(g)); bs.append(float(b))
    m = lambda a: float(np.mean(a))
    return {"y": m(ys), "ystd": m(ystd), "sat": m(ss),
            "r": m(rs), "g": m(gs), "b": m(bs),
            "rb": m(rs) / max(1e-6, m(bs)), "n": len(ys)}


def _rb_after_kelvin(in_pattern: str, kelvin: int) -> float:
    # apply colortemperature to the whole extracted frame set in one pass
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        subprocess.run(
            [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", in_pattern,
             "-vf", f"colortemperature=temperature={kelvin}:mix=1.0:pl=1.0",
             str(out / "k_%03d.png")],
            check=True,
        )
        return _stats_of(sorted(out.glob("k_*.png")))["rb"]


def _match_kelvin(in_pattern: str, target_rb: float) -> int:
    lo, hi = 3000, 9000
    for _ in range(14):
        m1 = lo + (hi - lo) // 3
        m2 = hi - (hi - lo) // 3
        e1 = abs(_rb_after_kelvin(in_pattern, m1) - target_rb)
        e2 = abs(_rb_after_kelvin(in_pattern, m2) - target_rb)
        if e1 < e2:
            hi = m2
        else:
            lo = m1
        if hi - lo <= 100:
            break
    return (lo + hi) // 2


def calibrate(raw: str, final: str) -> dict:
    p = DEFAULT_PARAMS
    with tempfile.TemporaryDirectory() as rd, tempfile.TemporaryDirectory() as fd:
        rdir, fdir = Path(rd), Path(fd)
        rframes = _extract(raw, rdir)
        fframes = _extract(final, fdir)
        R, F = _stats_of(rframes), _stats_of(fframes)
        prefix = rframes[0].name.split("_")[0]  # "f" or "g" fallback
        kelvin = _match_kelvin(str(rdir / f"{prefix}_%03d.png"), F["rb"])

    contrast_ratio = F["ystd"] / R["ystd"]
    sat_ratio = F["sat"] / R["sat"]
    k_contrast = (contrast_ratio - 1.0) / (p["contrast"] / 100.0)
    k_saturation = (sat_ratio - 1.0) / (p["saturation"] / 100.0)
    k_temp_kelvin = (6500.0 - kelvin) / p["temperature"] if p["temperature"] else 0.0

    return {
        "raw": R, "final": F,
        "contrast_ratio": contrast_ratio, "sat_ratio": sat_ratio,
        "matched_kelvin": kelvin,
        "derived": {
            "k_contrast": round(k_contrast, 3),
            "k_saturation": round(k_saturation, 3),
            "k_temp_kelvin": round(k_temp_kelvin, 2),
            "brightness_target": round(F["y"] / 255.0, 3),
        },
    }


def _write_config(derived: dict, path: Path) -> None:
    mp = dict(DEFAULT_MAP)
    mp["k_contrast"] = derived["k_contrast"]
    mp["k_saturation"] = derived["k_saturation"]
    mp["k_temp_kelvin"] = derived["k_temp_kelvin"]
    params = {"brightness_target": derived["brightness_target"]}
    path.write_text(json.dumps({"map": mp, "params": params},
                               ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None):
    argv = argv or sys.argv[1:]
    apply = "--apply" in argv
    pos = [a for a in argv if not a.startswith("--")]
    raw, final = pos[0], pos[1]
    res = calibrate(raw, final)
    R, F, d = res["raw"], res["final"], res["derived"]
    print(f"frames: raw={R['n']} final={F['n']}")
    print(f"  luma   Y: raw {R['y']:.1f}  final {F['y']:.1f}")
    print(f"  contr std: raw {R['ystd']:.1f}  final {F['ystd']:.1f}  ratio {res['contrast_ratio']:.3f}")
    print(f"  satur  S: raw {R['sat']:.1f}  final {F['sat']:.1f}  ratio {res['sat_ratio']:.3f}")
    print(f"  R/B ratio: raw {R['rb']:.3f}  final {F['rb']:.3f}  -> matched {res['matched_kelvin']}K")
    print("derived coefficients (for sliders 对比度+13 饱和度+8 色温-9):")
    for k, v in d.items():
        print(f"    {k}: {v}")
    if apply:
        out = ROOT / "calibration" / "calibrated.json"
        _write_config(d, out)
        print(f"\n✓ wrote {out}\n  use with:  --config {out.relative_to(ROOT)}")
    else:
        print("\n(run with --apply to write calibration/calibrated.json)")
    return res


if __name__ == "__main__":
    main()
