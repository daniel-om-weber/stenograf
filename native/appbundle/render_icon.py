"""Rasterize icon.svg into the two committed assets. Run through build.sh.

Every size is rendered from the vector rather than downscaled from one big
bitmap: at 16 and 32 px the waveform's rounded caps turn to mush when they are
resampled, and those are the sizes the menu bar and the Finder list use.

Qt does the rendering because PySide6 is already the GUI's dependency — no
ImageMagick, no Pillow, nothing new to install. Its renderer implements SVG
Tiny 1.2 only, so keep icon.svg inside that subset (no filters, no CSS).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRectF  # noqa: E402
from PySide6.QtGui import QGuiApplication, QImage, QPainter  # noqa: E402
from PySide6.QtSvg import QSvgRenderer  # noqa: E402

# (point size, scale) — the exact set `iconutil` expects in an .iconset.
ICONSET = [(size, scale) for size in (16, 32, 128, 256, 512) for scale in (1, 2)]
WINDOW_ICON = 512
"""Qt's window/Dock/taskbar icon, and the Linux .desktop entry's Icon=."""


def render(renderer: QSvgRenderer, pixels: int, path: Path) -> None:
    image = QImage(pixels, pixels, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(0)  # transparent: the squircle, not the canvas, is the icon
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter, QRectF(0, 0, pixels, pixels))
    painter.end()
    if not image.save(str(path), "PNG"):
        raise SystemExit(f"could not write {path}")


def main(source: Path, iconset: Path, window_icon: Path) -> None:
    QGuiApplication(sys.argv)  # QImage painting needs an application object
    renderer = QSvgRenderer(str(source))
    if not renderer.isValid():
        raise SystemExit(f"{source} is not valid SVG (Qt's parser rejects it)")

    iconset.mkdir(parents=True, exist_ok=True)
    for size, scale in ICONSET:
        suffix = "" if scale == 1 else f"@{scale}x"
        render(renderer, size * scale, iconset / f"icon_{size}x{size}{suffix}.png")
    window_icon.parent.mkdir(parents=True, exist_ok=True)
    render(renderer, WINDOW_ICON, window_icon)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
