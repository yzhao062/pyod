"""Build brand/team/core-team.png, the team card shown in the README.

From the repository root, with Pillow installed:

    python brand/team/build_team_image.py

Each entry in CORE and CONTRIBUTORS names its photo: a headshot in this folder,
a GitHub avatar downloaded at build time, or none, which draws an initials
badge in the PyOD palette. A failed avatar download stops the build before the
PNG is written. To change the team, edit the two lists and run the script
again.
"""
import io
import itertools
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE / "core-team.png"

CARDINAL = (153, 0, 0)   # #990000, the flagged outlier
NAVY = (43, 85, 151)     # #2b5597, the inlier distribution
INK = (31, 35, 40)
MUTED = (87, 96, 106)
RING = (225, 228, 232)

# (name, role, photo): photo is a file in this folder, "gh:<login>" for a
# GitHub avatar, or None for an initials badge.
CORE = [
    ("Yue Zhao", "Creator and lead · USC", "yue-zhao.jpg"),
    ("Xiyang Hu", "Co-lead · Arizona State", "xiyang-hu.jpg"),
    ("Zain Nasrullah", "PyOD 1 co-author", "gh:ZainNasrullah"),
    ("Winston (Zheng) Li", "PyOD 1 co-author", "winston-li.jpg"),
]
# The ten people with the most commits after the core team, aliases merged.
CONTRIBUTORS = [
    ("Yahya Almardeny", None),
    ("Akira Tamamori", None),
    ("Daniel Kulik", "gh:KulikDM"),
    ("Michiel Bongaerts", None),
    ("Zhuo Xiao", "gh:zhuox5"),
    ("Sihan Chen", "gh:Yeechin-is-here"),
    ("Roel Bouman", None),
    ("Yuehan Qin", "gh:yqin43"),
    ("Adam Goodge", "gh:agoodge"),
    ("Tiankai Yang", "gh:RaymondY"),
]
PER_ROW = 5

SCALE = 2          # draw at 2x; the README displays the image at half width
WIDTH = 900 * SCALE
PAD = 44 * SCALE
SUPER = 4          # supersampling factor for smooth circle edges


def font(size, bold=False):
    """Helvetica on macOS (face 1 of the collection is bold), DejaVu
    elsewhere (one face per file), then Pillow's built-in font."""
    face = 1 if bold else 0
    dejavu = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = [
        ("/System/Library/Fonts/HelveticaNeue.ttc", face),
        ("/System/Library/Fonts/Helvetica.ttc", face),
        (f"/usr/share/fonts/truetype/dejavu/{dejavu}", 0),
        (dejavu, 0),
    ]
    for path, index in candidates:
        try:
            return ImageFont.truetype(path, size, index=index)
        except (OSError, ValueError):
            continue
    return ImageFont.load_default(size=size)


def load_photo(photo):
    if photo is None:
        return None
    if photo.startswith("gh:"):
        url = f"https://github.com/{photo[3:]}.png?size=400"
        with urllib.request.urlopen(url, timeout=30) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
    return Image.open(HERE / photo).convert("RGB")


def circle(img, diameter):
    """Square-crop img and mask it to a circle with a thin ring."""
    big = diameter * SUPER
    side = min(img.size)
    left, top = (img.width - side) // 2, (img.height - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize(
        (big, big), Image.LANCZOS)
    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, big - 1, big - 1), fill=255)
    out = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    ring = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    ImageDraw.Draw(ring).ellipse((0, 0, big - 1, big - 1), outline=RING,
                                 width=3 * SCALE * SUPER)
    out = Image.alpha_composite(out, ring)
    return out.resize((diameter, diameter), Image.LANCZOS)


def badge(name, diameter, color):
    big = diameter * SUPER
    out = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(out)
    draw.ellipse((0, 0, big - 1, big - 1), fill=color)
    words = name.replace("(", "").replace(")", "").split()
    initials = (words[0][0] + words[-1][0]).upper()
    f = font(int(big * 0.36), bold=True)
    box = draw.textbbox((0, 0), initials, font=f)
    draw.text(((big - (box[2] - box[0])) / 2 - box[0],
               (big - (box[3] - box[1])) / 2 - box[1]),
              initials, font=f, fill=(255, 255, 255))
    return out.resize((diameter, diameter), Image.LANCZOS)


def centered(draw, cx, y, text, f, fill):
    """Draw text centered on cx; return the font's line height, so lines
    below start at the same y whatever letters this text contains."""
    width = draw.textlength(text, font=f)
    draw.text((cx - width / 2, y), text, font=f, fill=fill)
    ascent, descent = f.getmetrics()
    return ascent + descent


def label(draw, y, text):
    f = font(12 * SCALE, bold=True)
    spaced = " ".join(text.upper())
    centered(draw, WIDTH / 2, y, spaced, f, CARDINAL)
    return 12 * SCALE


def row_height(diameter, fonts):
    name_font, role_font = fonts
    return diameter + 10 * SCALE + (name_font.size + role_font.size) * 1.4


def row(layer, y, people, diameter, fonts, badge_colors):
    """Draw one row of (name, role, photo) entries; return its height."""
    draw = ImageDraw.Draw(layer)
    name_font, role_font = fonts
    slot = (WIDTH - 2 * PAD) / len(people)
    for i, (name, role, photo) in enumerate(people):
        cx = PAD + slot * (i + 0.5)
        img = load_photo(photo)
        disc = (badge(name, diameter, next(badge_colors)) if img is None
                else circle(img, diameter))
        layer.alpha_composite(disc, (int(cx - diameter / 2), int(y)))
        ty = y + diameter + 10 * SCALE
        h = centered(draw, cx, ty, name, name_font, INK)
        if role:
            centered(draw, cx, ty + h + 2 * SCALE, role, role_font, MUTED)
    return row_height(diameter, fonts)


def build():
    core_d, contrib_d = 104 * SCALE, 68 * SCALE
    core_fonts = (font(17 * SCALE, bold=True), font(13 * SCALE))
    contrib_fonts = (font(13 * SCALE, bold=True), font(12 * SCALE))
    n_rows = -(-len(CONTRIBUTORS) // PER_ROW)
    label_h = 12 * SCALE + 18 * SCALE
    # Size the card from the row count before drawing, so the rounded card
    # matches the content for any number of contributor rows.
    height = int(PAD + label_h + row_height(core_d, core_fonts) + 14 * SCALE
                 + label_h
                 + n_rows * (row_height(contrib_d, contrib_fonts) - 6 * SCALE)
                 + PAD - 18 * SCALE)

    mask = Image.new("L", (WIDTH * 2, height * 2), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, WIDTH * 2 - 1, height * 2 - 1), radius=28 * SCALE * 2, fill=255)
    mask = mask.resize((WIDTH, height), Image.LANCZOS)
    # A white card whose alpha is the mask, so edge pixels stay white
    # rather than blending toward black.
    card = Image.new("RGBA", (WIDTH, height), (255, 255, 255, 255))
    card.putalpha(mask)
    draw = ImageDraw.Draw(card)
    badge_colors = itertools.cycle([CARDINAL, NAVY])

    y = PAD
    y += label(draw, y, "Core team") + 18 * SCALE
    y += row(card, y, CORE, core_d, core_fonts, badge_colors) + 14 * SCALE
    y += label(draw, y, "Top contributors") + 18 * SCALE
    for start in range(0, len(CONTRIBUTORS), PER_ROW):
        chunk = [(name, None, photo)
                 for name, photo in CONTRIBUTORS[start:start + PER_ROW]]
        y += row(card, y, chunk, contrib_d, contrib_fonts,
                 badge_colors) - 6 * SCALE
    card.save(OUT, optimize=True)
    print(f"wrote {OUT} ({card.width}x{card.height})")


if __name__ == "__main__":
    build()
