"""Build the Movie Dubber icon set and splash (Pillow only).

The mark follows the Voxprint Audiobook Builder icon: one rounded-stroke glyph with a diagonal gradient on a dark
navy rounded tile.  Here the glyph is a sound wave inside a film frame (perforations top and bottom), violet instead
of the Audiobook Builder cyan.  16/24/32 px are drawn separately (wave only, heavier strokes) so they stay readable.

    python tools/make_icon.py                      icon set only
    python tools/make_icon.py --splash-art FILE    also the splash: FILE (square art) graded + the icon tile on top

Writes assets/voxprint-dubber.png (1024), assets/voxprint-dubber.ico and assets/voxprint-dubber-setup.ico
(16, 24, 32, 48, 64, 128, 256), installer/linux/voxprint-dubber-256.png and, with --splash-art, assets/splash.jpg
(1024 x 1024).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)
SMALL = (16, 24, 32)

CANVAS = (1, 13, 39)          # same navy as the Audiobook Builder icon
TILE = (7, 21, 56)
GRAD = ((216, 180, 254), (168, 85, 247), (109, 92, 255))   # lavender -> violet -> blue-violet, top-left to bottom-right
GLYPH_SCALE = 1.1             # 1024-unit design scaled up around the centre
SS = 4                        # supersampling factor


def _gradient(n: int) -> Image.Image:
    """Diagonal three-stop gradient, n x n."""
    small = Image.new("RGB", (256, 256))
    px = small.load()
    for y in range(256):
        for x in range(256):
            t = min(1.0, max(0.0, ((x + y) / 510 - 0.15) / 0.75))
            a, b, u = (GRAD[0], GRAD[1], t / 0.5) if t < 0.5 else (GRAD[1], GRAD[2], (t - 0.5) / 0.5)
            px[x, y] = tuple(round(a[i] + (b[i] - a[i]) * u) for i in range(3))
    return small.resize((n, n), Image.BICUBIC)


def _glyph_mask(n: int, small: bool) -> Image.Image:
    """White-on-black mask of the glyph in a 1024-unit design space, drawn at n px."""
    k = n / 1024
    m = Image.new("L", (n, n), 0)
    d = ImageDraw.Draw(m)

    def bar(x, h, w):
        r = w / 2
        d.rounded_rectangle(((x - r) * k, (512 - h / 2 - r) * k, (x + r) * k, (512 + h / 2 + r) * k), radius=r * k, fill=255)

    if small:                                   # wave only, five fat bars
        for x, h in zip((272, 392, 512, 632, 752), (150, 330, 470, 330, 150)):
            bar(x, h, 88)
        return m
    # film frame
    d.rounded_rectangle((196 * k, 228 * k, 828 * k, 796 * k), radius=96 * k, outline=255, width=round(34 * k))
    # perforations along the top and bottom of the frame
    for i in range(7):
        cx = 278 + i * 78
        for cy in (290, 734):
            d.rounded_rectangle(((cx - 20) * k, (cy - 17) * k, (cx + 20) * k, (cy + 17) * k), radius=9 * k, fill=255)
    # the sound wave
    for x, h in zip((290, 364, 438, 512, 586, 660, 734), (70, 150, 250, 300, 220, 130, 60)):
        bar(x, h, 40)
    return m


def icon(size: int) -> Image.Image:
    """The full icon at ``size`` px (opaque, like the Audiobook Builder icon)."""
    small = size in SMALL
    n = max(size, 256) * SS
    k = n / 1024
    img = Image.new("RGB", (n, n), CANVAS)
    pad = 24 if small else 40                                     # the tile nearly fills the square at small sizes
    tile = Image.new("L", (n, n), 0)
    ImageDraw.Draw(tile).rounded_rectangle((pad * k, pad * k, (1024 - pad) * k, (1024 - pad) * k), radius=176 * k, fill=255)
    img.paste(TILE, (0, 0, n, n), tile)
    mask = _glyph_mask(n, small)
    if not small:                                                 # the glyph fills the tile like the Audiobook Builder mark
        big = round(n * GLYPH_SCALE)
        off = (big - n) // 2
        mask = mask.resize((big, big), Image.LANCZOS).crop((off, off, off + n, off + n))
    grad = _gradient(n)
    if not small:                                                 # a soft glow under the strokes
        glow = mask.filter(ImageFilter.GaussianBlur(24 * k)).point(lambda v: v * 0.22)
        img.paste(grad, (0, 0), glow)
    img.paste(grad, (0, 0), mask)
    return img.resize((size, size), Image.LANCZOS).convert("RGBA")


def write_icons() -> None:
    ASSETS.mkdir(exist_ok=True)
    icon(1024).save(ASSETS / "voxprint-dubber.png", optimize=True)
    frames = [icon(s) for s in ICO_SIZES]
    for name in ("voxprint-dubber.ico", "voxprint-dubber-setup.ico"):
        frames[-1].save(ASSETS / name, format="ICO", sizes=[(s, s) for s in ICO_SIZES], append_images=frames[:-1])
    linux = ROOT / "installer" / "linux"
    linux.mkdir(parents=True, exist_ok=True)
    icon(256).save(linux / "voxprint-dubber-256.png", optimize=True)


def write_splash(art: Path) -> None:
    """Square art, graded towards a violet dusk, with the icon tile centred on it (same layout as Audiobook Builder)."""
    base = Image.open(art).convert("RGB")
    s = min(base.size)
    base = base.crop(((base.width - s) // 2, (base.height - s) // 2, (base.width + s) // 2, (base.height + s) // 2))
    base = base.resize((1024, 1024), Image.LANCZOS)
    base = ImageEnhance.Brightness(base).enhance(0.86)
    violet = Image.new("RGB", base.size, (88, 52, 150))
    sky = Image.linear_gradient("L").resize(base.size).point(lambda v: int(max(0, 200 - v) * 0.45))
    base = Image.composite(violet, base, sky)                     # stronger at the top (sky), fading out over the field
    tile = icon(1024).convert("RGB")
    t0, side = 236, 552                                           # the tile spans 236..788
    inner = tile.crop((40, 40, 984, 984)).resize((side, side), Image.LANCZOS)
    mask = Image.new("L", (side * SS, side * SS), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, side * SS - 1, side * SS - 1), radius=round(176 / 944 * side * SS), fill=255)
    mask = mask.resize((side, side), Image.LANCZOS)
    shadow = Image.new("L", base.size, 0)
    shadow.paste(mask, (t0, t0 + 14))
    shadow = shadow.filter(ImageFilter.GaussianBlur(22)).point(lambda v: int(v * 0.75))
    base.paste((0, 0, 8), (0, 0), shadow)
    base.paste(inner, (t0, t0), mask)
    base.save(ASSETS / "splash.jpg", quality=90, optimize=True, progressive=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--splash-art", type=Path, help="square artwork for the splash background")
    a = ap.parse_args()
    write_icons()
    if a.splash_art:
        write_splash(a.splash_art)
    print("written to", ASSETS)
