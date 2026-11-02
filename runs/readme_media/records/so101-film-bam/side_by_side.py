"""so101-film-bam: the same hero frames of two takes side by side (left 7v4_real, right 7v4_bam), one row per frame,
optionally cropped to a region, with a label. Pillow only.

    .venv/bin/python -I side_by_side.py <frames A> <frames B> <out.jpg> <f1,f2,...> [--crop x0,y0,x1,y1] [--width W]
"""
import os
import sys

from PIL import Image, ImageDraw

args = sys.argv[1:]
crop, width = None, 640
if "--crop" in args:
    i = args.index("--crop"); crop = tuple(int(v) for v in args[i + 1].split(",")); del args[i:i + 2]
if "--width" in args:
    i = args.index("--width"); width = int(args[i + 1]); del args[i:i + 2]
a, b, out, frames = args[0], args[1], args[2], [int(f) for f in args[3].split(",")]
rows = []
for f in frames:
    pair = []
    for d, lab in ((a, "7v4_real"), (b, "7v4_bam")):
        im = Image.open(os.path.join(d, f"frame_{f:04d}.png")).convert("RGB")
        if crop:
            im = im.crop(crop)
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        ImageDraw.Draw(im).text((8, 6), f"{lab}  frame {f}  t {f * 0.02:.2f} s", fill=(255, 255, 0))
        pair.append(im)
    row = Image.new("RGB", (2 * width + 6, pair[0].height), (255, 255, 255))
    row.paste(pair[0], (0, 0)); row.paste(pair[1], (width + 6, 0))
    rows.append(row)
sheet = Image.new("RGB", (rows[0].width, sum(r.height + 6 for r in rows)), (255, 255, 255))
y = 0
for r in rows:
    sheet.paste(r, (0, y)); y += r.height + 6
sheet.save(out, quality=90)
print(out, sheet.size)
