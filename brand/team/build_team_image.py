"""Build brand/team/core-team.png, the team card shown in the README.

Run from anywhere with Pillow installed:

    python brand/team/build_team_image.py

Yue Zhao, Xiyang Hu, and Winston (Zheng) Li use the headshots in this folder.
Everyone else uses their GitHub avatar, downloaded at build time; a person
without a usable photo gets an initials badge in the PyOD palette. To change
the team, edit CORE and CONTRIBUTORS below and run the script again.
"""
import io
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

# (name, role, photo): photo is a file in this folder, a GitHub login, or
# None for an initials badge.
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

BADGE_COLORS = [CARDINAL, NAVY]
BADGES_DRAWN = []

SCALE = 2          # draw at 2x; the README displays the image at half width
WIDTH = 900 * SCALE
PAD = 44 * SCALE
SUPER = 4          # supersampling factor for smooth circle edges


def font(size, bold=False):
    candidates = (
        ["/System/Library/Fonts/HelveticaNeue.ttc",
         "/System/Library/Fonts/Helvetica.ttc"]
        if not bold else
        ["/System/Library/Fonts/HelveticaNeue.ttc",
         "/System/Library/Fonts/Helvetica.ttc"])
    candidates += ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for path in candidates:
        try:
            # index 1 of Helvetica Neue / Helvetica is the bold face
            return ImageFont.truetype(path, size, index=1 if bold else 0)
        except (OSError, ValueError):
            continue
    return ImageFont.load_default()


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
    words = [w for w in name.replace("(", "").replace(")", "").split()]
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


def row(canvas, draw, y, people, diameter, name_font, role_font):
    n = len(people)
    slot = (WIDTH - 2 * PAD) / n
    for i, person in enumerate(people):
        name, role, photo = person if len(person) == 3 else (*person[:1],
                                                             None, person[1])
        cx = PAD + slot * (i + 0.5)
        img = load_photo(photo)
        if img is None:
            disc = badge(name, diameter, BADGE_COLORS[len(BADGES_DRAWN) % 2])
            BADGES_DRAWN.append(name)
        else:
            disc = circle(img, diameter)
        canvas.alpha_composite(disc, (int(cx - diameter / 2), int(y)))
        ty = y + diameter + 10 * SCALE
        h = centered(draw, cx, ty, name, name_font, INK)
        if role:
            centered(draw, cx, ty + h + 2 * SCALE, role, role_font, MUTED)
    return diameter + 10 * SCALE + (name_font.size + role_font.size) * 1.4


def build():
    core_d, contrib_d = 104 * SCALE, 68 * SCALE
    per_row = 5
    height = int(PAD + 12 * SCALE + 18 * SCALE + core_d + 60 * SCALE
                 + 12 * SCALE + 18 * SCALE
                 + 2 * (contrib_d + 40 * SCALE) + PAD)
    canvas = Image.new("RGBA", (WIDTH, height), (0, 0, 0, 0))
    card = Image.new("L", (WIDTH * 2, height * 2), 0)
    ImageDraw.Draw(card).rounded_rectangle(
        (0, 0, WIDTH * 2 - 1, height * 2 - 1), radius=28 * SCALE * 2, fill=255)
    card = card.resize((WIDTH, height), Image.LANCZOS)
    canvas.paste(Image.new("RGBA", (WIDTH, height), (255, 255, 255, 255)),
                 (0, 0), card)
    draw = ImageDraw.Draw(canvas)

    y = PAD
    y += label(draw, y, "Core team") + 18 * SCALE
    y += row(canvas, draw, y, CORE, core_d, font(17 * SCALE, bold=True),
             font(13 * SCALE)) + 14 * SCALE
    y += label(draw, y, "Top contributors") + 18 * SCALE
    for start in range(0, len(CONTRIBUTORS), per_row):
        chunk = [(name, None, photo)
                 for name, photo in CONTRIBUTORS[start:start + per_row]]
        y += row(canvas, draw, y, chunk, contrib_d,
                 font(13 * SCALE, bold=True), font(12 * SCALE)) - 6 * SCALE
    canvas = canvas.crop((0, 0, WIDTH, int(y + PAD - 18 * SCALE)))
    canvas.save(OUT, optimize=True)
    print(f"wrote {OUT} ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    build()
