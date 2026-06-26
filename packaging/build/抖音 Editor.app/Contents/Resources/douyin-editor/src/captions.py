"""
Caption renderer: turn (start, end, text) segments into full-frame transparent
PNG overlays styled like 抖音 captions, then build the ffmpeg overlay chain.

Why not libass: this ffmpeg's libass can't rasterize Apple's *color* emoji, so
出🚗 came out as a tofu box. We render captions ourselves with Pillow, mixing a
CJK font (text, white fill + black stroke) with Apple Color Emoji (color glyph,
rendered at a valid sbix strike and scaled). PC / B力 / X复 are plain text.
"""
from __future__ import annotations
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .fonts import cjk_font, emoji_font

CJK_FONT = cjk_font()
EMOJI_FONT, EMOJI_STRIKE = emoji_font()

# Codepoints we render via the color-emoji font rather than the CJK font.
def _is_emoji(ch: str) -> bool:
    o = ord(ch)
    return (
        0x1F000 <= o <= 0x1FAFF   # pictographs, transport (🚗 = U+1F697), symbols
        or 0x2600 <= o <= 0x27BF  # misc symbols / dingbats
        or o in (0x2B50, 0x2764)
    )


class CaptionStyle:
    def __init__(self, width=1080, height=1920):
        self.width = width
        self.height = height
        self.fontsize = round(height * 0.037)        # ~71px
        self.stroke = max(3, round(self.fontsize * 0.09))
        self.line_gap = round(self.fontsize * 0.30)
        self.margin_v = round(height * 0.13)         # baseline area above bottom
        self.max_text_w = round(width * 0.86)
        self.fill = (255, 255, 255, 255)
        self.stroke_fill = (0, 0, 0, 255)


class _Fonts:
    def __init__(self, size: int):
        self.cjk = ImageFont.truetype(CJK_FONT, size)
        self.emoji_raw = ImageFont.truetype(EMOJI_FONT, EMOJI_STRIKE)
        self.emoji_px = round(size * 1.0)
        self._emoji_cache: dict[str, Image.Image] = {}

    def char_width(self, ch: str) -> int:
        if _is_emoji(ch):
            return self.emoji_px
        return round(self.cjk.getlength(ch))

    def emoji_image(self, ch: str) -> Image.Image:
        if ch not in self._emoji_cache:
            tmp = Image.new("RGBA", (EMOJI_STRIKE * 2, EMOJI_STRIKE * 2), (0, 0, 0, 0))
            d = ImageDraw.Draw(tmp)
            d.text((EMOJI_STRIKE // 2, EMOJI_STRIKE // 2), ch,
                   font=self.emoji_raw, embedded_color=True)
            bbox = tmp.getbbox()
            glyph = tmp.crop(bbox) if bbox else tmp
            glyph = glyph.resize((self.emoji_px, self.emoji_px), Image.LANCZOS)
            self._emoji_cache[ch] = glyph
        return self._emoji_cache[ch]


TAIL_MAX = 4          # if the overflow tail is <= this many words, shrink to fit
MIN_SHRINK_SCALE = 0.80  # don't shrink a line below this (else split instead)
MIN_DUR = 0.5         # min seconds a caption stays on screen (hold short ones)
SENT_END = set("。.！!？?；;…⋯")  # strong sentence enders -> always start a new caption
GAP_BREAK = 0.30      # a pause longer than this (s) between words = phrase boundary


def _phrase_units(seg: list) -> list[list]:
    """Split one ASR segment into phrase units at sentence punctuation and at
    pauses between words, so a caption never glues the start of the next sentence
    onto the previous one (e.g. the '是' after '什么？', or a new phrase '研究')."""
    units, cur = [], []
    prev_end, prev_is_end = None, False
    for s, e, word in seg:
        gap = (s - prev_end) if prev_end is not None else 0.0
        if cur and (prev_is_end or gap > GAP_BREAK):
            units.append(cur); cur = []
        cur.append((s, e, word))
        prev_end = e
        w = word.strip()
        prev_is_end = bool(w) and w[-1] in SENT_END
    if cur:
        units.append(cur)
    return units


def build_caption_items(segments: list[list], style: CaptionStyle,
                        transform=lambda t: t) -> list[dict]:
    """Turn word-timestamped ASR segments into one-line caption items with
    accurate timing. `transform` produces the on-screen text (punctuation strip
    + word swaps) and is also used for width measurement.

    One caption per ASR segment (a natural breath/sentence unit). If a segment is
    too wide for one line: when only a small tail (<= TAIL_MAX words) overflows,
    keep it as ONE line and shrink the font slightly to fit; otherwise split into
    several one-line captions at word boundaries.

    Returns [{start, end, text, scale}, ...].
    """
    fonts = _Fonts(style.fontsize)
    max_w = style.max_text_w

    def wof(t: str) -> int:
        return sum(fonts.char_width(c) for c in t if not c.isspace())

    def joined(words) -> str:
        return transform("".join(w[2].strip() for w in words))

    # break each ASR segment into phrase units (sentence punctuation + pauses)
    units = []
    for seg in segments:
        units.extend(_phrase_units(seg))

    items = []
    for seg in units:
        disp = joined(seg)
        if not disp:
            continue
        if wof(disp) <= max_w:
            items.append({"start": seg[0][0], "end": seg[-1][1], "text": disp, "scale": 1.0})
            continue

        # how many words fit on the first line at base size?
        line_n = 1
        for i in range(1, len(seg) + 1):
            if wof(joined(seg[:i])) > max_w:
                line_n = max(1, i - 1)
                break
        tail_n = len(seg) - line_n
        fit_scale = max_w / wof(disp)

        if 0 < tail_n <= TAIL_MAX and fit_scale >= MIN_SHRINK_SCALE:
            # keep as one line, slightly smaller font
            items.append({"start": seg[0][0], "end": seg[-1][1],
                          "text": disp, "scale": round(fit_scale, 3)})
        else:
            # split into several one-line captions, BALANCED so the last line
            # isn't a lonely fragment (e.g. a single 慢). Aim for N even lines.
            base_items = len(items)
            full_w = wof(disp)
            n_lines = max(2, math.ceil(full_w / max_w))
            target = full_w / n_lines
            ww = [wof(joined([w])) for w in seg]

            i = 0
            while i < len(seg):
                j, cur, cur_w = i, "", 0
                while j < len(seg):
                    nxt = cur_w + ww[j]
                    # hard cap at one line; soft cap at the balanced target
                    over_hard = j > i and wof(joined(seg[i:j + 1])) > max_w
                    over_soft = (j > i and cur_w >= target
                                 and (len(items) - base_items) < n_lines - 1)
                    if over_hard or over_soft:
                        break
                    cur = joined(seg[i:j + 1])
                    cur_w = nxt
                    j += 1
                j = max(j, i + 1)
                cw = seg[i:j]
                if cur:
                    items.append({"start": cw[0][0], "end": cw[-1][1],
                                  "text": cur, "scale": 1.0})
                i = j

    # hold very short captions on screen longer (into the following silence),
    # without overlapping the next caption -> no flicker, no desync
    for k, it in enumerate(items):
        if it["end"] - it["start"] < MIN_DUR:
            limit = items[k + 1]["start"] - 0.05 if k + 1 < len(items) else it["end"] + MIN_DUR
            it["end"] = max(it["end"], min(it["start"] + MIN_DUR, limit))
    return items


def render_caption_png(text: str, style: CaptionStyle, out_path: str,
                       scale: float = 1.0) -> None:
    """Render a single-line caption onto a full-frame transparent PNG. `scale`
    shrinks the font (used for the shrink-to-fit tail rule); a safety fit keeps
    the line within the frame regardless."""
    fontsize = max(8, round(style.fontsize * scale))
    fonts = _Fonts(fontsize)
    stroke = max(2, round(style.stroke * scale))

    # safety: never let a line exceed the usable width
    line_w = sum(fonts.char_width(c) for c in text)
    if line_w > style.max_text_w:
        fontsize = max(8, round(fontsize * style.max_text_w / line_w))
        fonts = _Fonts(fontsize)
        stroke = max(2, round(style.stroke * fontsize / style.fontsize))
        line_w = sum(fonts.char_width(c) for c in text)

    img = Image.new("RGBA", (style.width, style.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    x = (style.width - line_w) // 2
    # keep the baseline area constant: anchor the (full-size) line bottom
    y = style.height - style.margin_v - fontsize
    for ch in text:
        if _is_emoji(ch):
            glyph = fonts.emoji_image(ch)
            gy = y + (fontsize - glyph.height) // 2
            img.alpha_composite(glyph, (x, max(0, gy)))
            x += fonts.emoji_px
        else:
            draw.text((x, y), ch, font=fonts.cjk, fill=style.fill,
                      stroke_width=stroke, stroke_fill=style.stroke_fill)
            x += fonts.char_width(ch)
    img.save(out_path)


def render_items(items: list[dict], style: CaptionStyle, out_dir: str) -> list[dict]:
    """Render one PNG per caption item ({start, end, text, scale})."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rendered = []
    for i, it in enumerate(items):
        p = str(out / f"cap_{i:04d}.png")
        render_caption_png(it["text"], style, p, scale=it.get("scale", 1.0))
        rendered.append({"png": p, "start": it["start"], "end": it["end"]})
    return rendered
