"""Generate the application icon (assets/voxprint-dubber.ico + .png): a dark rounded square with a violet sound-wave.  Needs Pillow."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "assets"


def draw(size: int = 512) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, size - 1, size - 1), radius=size // 5, fill=(27, 27, 35, 255), outline=(124, 92, 230, 255), width=size // 40)
    heights = [0.18, 0.34, 0.56, 0.74, 0.5, 0.64, 0.3, 0.2]
    n, pad = len(heights), size * 0.17
    step = (size - 2 * pad) / n
    for i, h in enumerate(heights):
        x = pad + i * step + step * 0.15
        w = step * 0.7
        hh = size * h * 0.8
        col = (160, 130, 255, 255) if i % 2 == 0 else (124, 92, 230, 255)
        d.rounded_rectangle((x, size / 2 - hh / 2, x + w, size / 2 + hh / 2), radius=w / 2, fill=col)
    return img


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    big = draw(512)
    big.save(OUT / "voxprint-dubber.png")
    big.save(OUT / "voxprint-dubber.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("icons written to", OUT)
