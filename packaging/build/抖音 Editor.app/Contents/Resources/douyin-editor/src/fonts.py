"""
Cross-platform font resolution for captions.

Local macOS uses Apple's system fonts (what the operator approved). On Linux /
Render — or when FORCE_BUNDLED_FONTS=1 — it uses the bundled Noto fonts so the
container renders identical captions (incl. the color 🚗 emoji). Env overrides
let the deployment pin exact files.

Color-emoji fonts only have fixed bitmap strikes: Apple Color Emoji = 160px,
Noto Color Emoji = 109px. The resolver returns the right strike with the path.
"""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FONTS = ROOT / "assets" / "fonts"

_BUNDLED_CJK = FONTS / "NotoSansCJKsc-Medium.otf"
_BUNDLED_EMOJI = (str(FONTS / "NotoColorEmoji.ttf"), 109)
# Apple fonts are a last-resort fallback only; bundled Noto is preferred so local
# and cloud (Linux) output are identical.
_APPLE_CJK = "/System/Library/Fonts/STHeiti Medium.ttc"
_APPLE_EMOJI = ("/System/Library/Fonts/Apple Color Emoji.ttc", 160)


def _exists(p) -> bool:
    return bool(p) and Path(p).exists()


def cjk_font() -> str:
    for c in [os.environ.get("CJK_FONT"), _BUNDLED_CJK, _APPLE_CJK,
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"]:
        if _exists(c):
            return str(c)
    raise FileNotFoundError("no CJK font found (set CJK_FONT or bundle Noto)")


def emoji_font() -> tuple[str, int]:
    """Return (path, bitmap_strike). Strike is fixed per font: Noto=109, Apple=160."""
    env = os.environ.get("EMOJI_FONT")
    if _exists(env):
        return env, int(os.environ.get("EMOJI_STRIKE", "109"))
    for p, strike in [_BUNDLED_EMOJI, _APPLE_EMOJI]:
        if _exists(p):
            return p, strike
    raise FileNotFoundError("no color-emoji font found (set EMOJI_FONT or bundle Noto)")
