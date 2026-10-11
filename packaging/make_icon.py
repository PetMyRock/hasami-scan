"""Draw the app icon (a flatbed scanner) as a multi-size .ico: python make_icon.py OUT.ico"""

import sys

from PIL import Image, ImageDraw


def draw(size: int) -> Image.Image:
    s = size / 256
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([8 * s, 8 * s, 248 * s, 248 * s], radius=48 * s, fill=(44, 44, 52, 255))
    # Lid (open, tilted back) and the scanner's body.
    d.polygon([(40 * s, 120 * s), (216 * s, 120 * s), (196 * s, 52 * s), (60 * s, 52 * s)], fill=(214, 214, 222, 255))
    d.rounded_rectangle([32 * s, 128 * s, 224 * s, 208 * s], radius=14 * s, fill=(236, 236, 242, 255))
    # Glass and the light bar.
    d.rectangle([48 * s, 140 * s, 208 * s, 176 * s], fill=(70, 74, 88, 255))
    d.rectangle([118 * s, 136 * s, 132 * s, 180 * s], fill=(255, 112, 67, 255))
    d.ellipse([190 * s, 188 * s, 204 * s, 202 * s], fill=(255, 112, 67, 255))
    return im


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "hasami.ico"
    sizes = [16, 24, 32, 48, 64, 128, 256]
    big = draw(256)
    big.save(out, sizes=[(n, n) for n in sizes], append_images=[draw(n) for n in sizes[:-1]])
    print(out)
