#!/usr/bin/env python3
"""
mandelbrot-codex.py

Interactive Mandelbrot set viewer with click-and-drag zoom and reset.

- Drag to draw a rectangle and zoom into that region.
- Click (without dragging) to reset to the default view.
- Press 'r' to reset the view.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.backend_bases import MouseButton
from infrastructure.logger import MandelbrotLogger
import os
from infrastructure.config import (
    WIDTH,
    HEIGHT,
    MAX_ITER,
    DEFAULT_VIEW,
    FIGURE_SIZE,
    COLORMAP,
)

# Initialize view and figure
view = list(DEFAULT_VIEW)
fig, ax = plt.subplots(figsize=FIGURE_SIZE)
logger = MandelbrotLogger(__file__)


def mandelbrot(re_min, re_max, im_min, im_max):
    re = np.linspace(re_min, re_max, WIDTH)
    im = np.linspace(im_min, im_max, HEIGHT)
    c = re[np.newaxis, :] + 1j * im[:, np.newaxis]
    z = np.zeros_like(c)
    m = np.full(c.shape, MAX_ITER, dtype=int)
    mask = np.ones(c.shape, dtype=bool)
    for i in range(MAX_ITER):
        z[mask] = z[mask] * z[mask] + c[mask]
        mask_new = np.abs(z) <= 2
        m[mask & ~mask_new] = i
        mask = mask & mask_new
        if not mask.any():
            break
    return m


def draw():
    ax.clear()
    fig.suptitle(os.path.basename(__file__), fontsize=14, fontweight="bold")
    with logger.timeit("Compute Mandelbrot"):
        m = mandelbrot(*view)
    ax.imshow(
        m, extent=(view[0], view[1], view[2], view[3]), cmap=COLORMAP, origin="lower"
    )
    ax.set_title(
        f"Mandelbrot Set (drag to zoom, click to reset)\nResolution: {WIDTH}x{HEIGHT}, Max Iterations: {MAX_ITER}"
    )
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    plt.draw()


def on_select(eclick, erelease):
    x0, y0 = eclick.xdata, eclick.ydata
    x1, y1 = erelease.xdata, erelease.ydata
    if abs(x1 - x0) < 1e-6 and abs(y1 - y0) < 1e-6:
        view[:] = DEFAULT_VIEW
    else:
        view[:] = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
    draw()


def main():
    draw()  # Draw before creating selector!
    selector = RectangleSelector(
        ax,
        on_select,
        useblit=True,
        button=[MouseButton.LEFT],
        spancoords="data",
        interactive=False,
        minspanx=0,
        minspany=0,
    )
    plt.show()


if __name__ == "__main__":
    main()
