"""
Smart HDR export — wrap the graded SDR picture into a proper HDR10 container.

The operator's footage is SDR (bt709); 剪映's "智能HDR" is essentially an
SDR->HDR expansion: linearize from bt709, then re-encode to BT.2020 primaries +
SMPTE-2084 (PQ) as 10-bit HEVC with HDR10 metadata. We don't fabricate detail
that isn't there — we map SDR reference white (100 nits) into the HDR container
so it's tagged/displayed as HDR. True HDR source (10-bit) would expand further.
"""
from __future__ import annotations

# Appended AFTER grade + caption overlays, before encode. The input is forced to
# bt709 because the eq/overlay chain (and the libx264 beauty intermediate) leave
# the frame's colour tags undefined -> zscale would error "no path between
# colorspaces" without explicit input characteristics.
HDR_CONVERT = (
    "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv,"
    "zscale=t=linear:npl=100,format=gbrpf32le,"
    "zscale=p=bt2020:t=smpte2084:m=bt2020nc:r=tv,format=yuv420p10le"
)

_X265_PARAMS = (
    "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:hdr10-opt=1:"
    "master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):"
    "max-cll=1000,400"
)


def hdr_encode_args(export: dict) -> list[str]:
    return [
        "-c:v", "libx265",
        "-pix_fmt", "yuv420p10le",
        "-preset", export["preset"],
        "-crf", str(export.get("hdr_crf", 20)),
        "-x265-params", _X265_PARAMS,
        "-tag:v", "hvc1",
        "-r", str(export["fps"]),
        "-c:a", export["audio_codec"],
        "-b:a", export["audio_bitrate"],
        "-movflags", "+faststart",
    ]
