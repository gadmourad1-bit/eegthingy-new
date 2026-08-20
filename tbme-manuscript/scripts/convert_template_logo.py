#!/usr/bin/env python3
"""Deterministically convert the official indexed-color EPS logo to PNG.

The vendor EPS contains one uncompressed indexed raster.  Parsing that raster
directly avoids a Ghostscript dependency and keeps the conversion reproducible.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "template" / "LOGO-generic-web.eps"
DEFAULT_OUTPUT = ROOT / "template" / "LOGO-generic-web.png"


def convert(source: Path, output: Path) -> None:
    text = source.read_text(encoding="ascii")
    palette_match = re.search(
        r"\[/Indexed\s+/DeviceRGB\s+255\s*<([0-9A-Fa-f\s]+)>\s*\]\s*setcolorspace",
        text,
        flags=re.DOTALL,
    )
    geometry_match = re.search(
        r"/Width\s+(\d+)\s+/Height\s+(\d+).*?\nimage\s*\n",
        text,
        flags=re.DOTALL,
    )
    if palette_match is None or geometry_match is None:
        raise SystemExit("unsupported vendor EPS structure")

    data_start = geometry_match.end()
    data_match = re.match(r"([0-9A-Fa-f\s]+)>\s*showpage", text[data_start:], re.DOTALL)
    if data_match is None:
        raise SystemExit("vendor EPS raster payload is missing")

    width, height = (int(value) for value in geometry_match.groups())
    palette = bytes.fromhex(palette_match.group(1))
    pixels = bytes.fromhex(data_match.group(1))
    if len(palette) != 256 * 3:
        raise SystemExit(f"unexpected palette length: {len(palette)}")
    if len(pixels) != width * height:
        raise SystemExit(
            f"unexpected raster length: {len(pixels)}; expected {width * height}"
        )

    image = Image.frombytes("P", (width, height), pixels)
    image.putpalette(palette)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=False, compress_level=9)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    convert(args.source.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
