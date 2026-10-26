"""Generate a synthetic tail-suspension style test clip with two "mice".

Each mouse is a dark ellipse (body) with a head bump, built with the `geq`
alpha mask and composited over a light background. One mouse moves around,
the other stays (mostly) still - this is enough for YOLO11n to fire on the
COCO `mouse` class and exercises the tracker's identity handling.
"""
import subprocess
import sys

out = sys.argv[1] if len(sys.argv) > 1 else "media/synthetic_mice.mp4"
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
fps = 25
W, H = 960, 540

# --- helpers ---------------------------------------------------------------
def ellipse(sw, sh, color, rx, ry):
    return (
        f"color=c={color}:s={sw}x{sh}:r={fps},format=rgba,"
        f"geq=r='{int(color, 16) >> 16 & 255}':g='{int(color, 16) >> 8 & 255}':"
        f"b='{int(color, 16) & 255}':"
        f"a='255*lt(pow((X-{sw / 2:.1f})/{rx:.1f},2)+pow((Y-{sh / 2:.1f})/{ry:.1f},2),1)'"
    )

def bg():
    return f"color=c=0xD8D8D0:s={W}x{H}:r={fps}"

# --- mouse A (moving) -----------------------------------------------------
body_a = ellipse(130, 56, "0x2E2C2A", 62, 26)
head_a = ellipse(46, 30, "0x363330", 21, 13)
tail_a = ellipse(150, 9, "0x2E2C2A", 74, 4)
# --- mouse B (still) ------------------------------------------------------
body_b = ellipse(126, 54, "0x363330", 60, 25)
head_b = ellipse(44, 28, "0x3E3B37", 20, 12)
tail_b = ellipse(140, 9, "0x363330", 69, 4)

# moving mouse follows a slow figure-ish path
ax = f"250+170*sin(2*PI*t/{dur})"
ay = f"190+45*sin(4*PI*t/{dur})"

fc = ";".join(
    [
        f"{bg()}[bg]",
        f"{body_a}[bodyA]",
        f"{head_a}[headA]",
        f"{tail_a}[tailA]",
        f"{body_b}[bodyB]",
        f"{head_b}[headB]",
        f"{tail_b}[tailB]",
        # moving mouse
        f"[bg][tailA]overlay=x='{ax}+55':y='{ay}+52'[m1]",
        f"[m1][bodyA]overlay=x='{ax}':y='{ay}'[m2]",
        f"[m2][headA]overlay=x='{ax}+105':y='{ay}+6'[m3]",
        # still mouse
        "[m3][tailB]overlay=x=742:y=396[s1]",
        "[s1][bodyB]overlay=x=660:y=345[s2]",
        "[s2][headB]overlay=x=762:y=352[out]",
    ]
)

cmd = [
    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
    "-filter_complex", fc,
    "-map", "[out]", "-t", str(dur),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out,
]
subprocess.run(cmd, check=True)
print("wrote", out)
