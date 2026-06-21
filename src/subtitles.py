"""
Subtitle pipeline: Chinese ASR -> word swaps -> Douyin-style ASS for burn-in.

    transcribe()  faster-whisper -> [(start, end, text), ...]
    apply_swaps() censorship-evasion replacements (嫖娼->PC etc.)
    to_ass()      phrase-level captions styled like a 抖音 video

Burn-in itself happens in the grade chain via the `subtitles` filter, so the
text sits on top of the final graded picture.
"""
from __future__ import annotations
import unicodedata
from pathlib import Path

from .config import WORD_SWAPS

# Question marks are the only punctuation we keep on screen.
KEEP_PUNCT = {"?", "？"}
# Characters that end a phrase: drop these (break the line), but ?？ break AND stay.
BREAK_DROP = set("，,。．.！!；;：:、…⋯\n　 ")


def strip_punct(text: str) -> str:
    """Remove all punctuation except ? / ？."""
    out = []
    for ch in text:
        if ch in KEEP_PUNCT:
            out.append(ch)
        elif unicodedata.category(ch).startswith("P"):
            continue
        else:
            out.append(ch)
    return "".join(out).strip()


def split_segment(start: float, end: float, text: str) -> list[tuple]:
    """Split one ASR segment into punctuation-delimited phrases, strip punctuation
    (keeping ?/？), and spread the segment's time across them by length."""
    phrases, cur = [], ""
    for ch in text:
        if ch in KEEP_PUNCT:
            cur += ch
            phrases.append(cur); cur = ""
        elif ch in BREAK_DROP:
            if cur.strip():
                phrases.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        phrases.append(cur)

    phrases = [p for p in (strip_punct(p) for p in phrases) if p]
    if not phrases:
        return []
    total = sum(len(p) for p in phrases)
    out, t = [], start
    for p in phrases:
        dur = (end - start) * len(p) / total
        out.append((t, t + dur, p))
        t += dur
    return out


def transcribe(src: str, model_size: str = "small", language: str = "zh",
               device: str = "cpu", compute_type: str = "int8") -> list[tuple]:
    """Return [(start_sec, end_sec, text), ...]. First call downloads the model."""
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(
        src, language=language, vad_filter=True,
        # condition_on_previous_text off reduces runaway repetition on short clips
        condition_on_previous_text=False,
    )
    out = []
    for s in segments:
        text = s.text.strip()
        if text:
            out.append((s.start, s.end, text))
    return out


def transcribe_words(src: str, model_size: str = "small", language: str = "zh",
                     device: str = "cpu", compute_type: str = "int8") -> list[list]:
    """Return words grouped by ASR segment: [[(start,end,word), ...], ...].

    Word-level timestamps give accurate timing (segment timestamps drift ~2s);
    keeping the segment grouping preserves natural breath/sentence units so
    captions don't break mid-phrase.
    """
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(
        src, language=language, vad_filter=True,
        condition_on_previous_text=False, word_timestamps=True,
    )
    out = []
    for s in segments:
        seg = []
        if s.words:
            for w in s.words:
                if w.word.strip():
                    seg.append((w.start, w.end, w.word.strip()))
        elif s.text.strip():  # fallback if word timing missing
            seg.append((s.start, s.end, s.text.strip()))
        if seg:
            out.append(seg)
    return out


# Single characters that must not start a caption line (aspect/structural
# particles, modal finals): merge them back onto the preceding word so a word
# like 盯着 / 看了 / 没用的 is never split across captions.
_TRAILING = set("着了过的地得们吗呢吧啊呀嘛啦哦呗喽嘞的")


def _merge_particles(words: list[str]) -> list[str]:
    out = []
    for w in words:
        if out and len(w) == 1 and w in _TRAILING:
            out[-1] += w
        else:
            out.append(w)
    return out


def to_word_units(segments: list[list]) -> list[list]:
    """Convert per-character ASR tokens into Chinese *word* units (jieba) so that
    caption breaks can only fall between words, never inside one (e.g. 盯着, 手机).

    Builds a char-level timeline from the word-timestamped tokens, segments the
    text with jieba, merges trailing particles, and maps each word back to the
    timing of its first/last character.
    """
    import jieba
    jieba.setLogLevel(60)
    out = []
    for seg in segments:
        chars = []
        for s, e, tok in seg:
            tok = tok.strip()
            if not tok:
                continue
            n = len(tok)
            for i, ch in enumerate(tok):
                chars.append((s + (e - s) * i / n, s + (e - s) * (i + 1) / n, ch))
        if not chars:
            continue
        text = "".join(c[2] for c in chars)
        words = _merge_particles(list(jieba.cut(text, HMM=True)))
        units, idx = [], 0
        for w in words:
            wc = chars[idx:idx + len(w)]
            idx += len(w)
            if w.strip() and wc:
                units.append((wc[0][0], wc[-1][1], w))
        if units:
            out.append(units)
    return out


def apply_swaps(text: str, swaps: dict | None = None) -> str:
    swaps = swaps or WORD_SWAPS
    for src, dst in swaps.items():
        text = text.replace(src, dst)
    return text


def _ass_time(t: float) -> str:
    if t < 0:
        t = 0
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def to_ass(segments: list[tuple], width: int = 1080, height: int = 1920,
           font: str = "PingFang SC", swaps: dict | None = None) -> str:
    """Build an ASS document with 抖音-style bottom captions."""
    # Sizing relative to a 1920-tall frame; scales with height.
    fontsize = round(height * 0.037)        # ~71px at 1080x1920
    outline = max(2, round(height * 0.0022)) # ~4px
    margin_v = round(height * 0.13)          # ~12-13% up from bottom

    style = (
        "Style: Default,{font},{fs},&H00FFFFFF,&H000000FF,&H00000000,"
        "&H64000000,-1,0,0,0,100,100,0,0,1,{outline},0,2,90,90,{mv},1"
    ).format(font=font, fs=fontsize, outline=outline, mv=margin_v)

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\n"
        "WrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
        "OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,"
        "ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,"
        "MarginL,MarginR,MarginV,Encoding\n"
        f"{style}\n\n"
        "[Events]\n"
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )

    lines = []
    for start, end, text in segments:
        text = apply_swaps(text, swaps).replace("\n", " ")
        lines.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{text}"
        )
    return header + "\n".join(lines) + "\n"


def write_ass(src: str, dst_ass: str, model_size: str = "small",
              width: int = 1080, height: int = 1920,
              font: str = "PingFang SC", swaps: dict | None = None) -> list[tuple]:
    """Transcribe src and write an ASS file. Returns the (swapped) segments."""
    segs = transcribe(src, model_size=model_size)
    ass = to_ass(segs, width=width, height=height, font=font, swaps=swaps)
    Path(dst_ass).write_text(ass, encoding="utf-8")
    swapped = [(s, e, apply_swaps(t, swaps)) for s, e, t in segs]
    return swapped
