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
from logger import MandelbrotLogger
import os

logger = MandelbrotLogger(__file__)

width, height = 800, 600  # Standardized for benchmarking
max_iter = 100  # Standardized for benchmarking


class MandelbrotViewer:
    """Interactive viewer for the Mandelbrot set."""

    def __init__(self, width=800, height=600, max_iter=200, cmap="hot"):
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.cmap = cmap
        self.default_view = (-2.0, 1.0, -1.0, 1.0)
        self.view = self.default_view

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.im = None

        self.selector = RectangleSelector(
            self.ax,
            self._on_select,
            useblit=True,
            button=[MouseButton.LEFT],
            spancoords="data",
            interactive=False,
        )
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._draw()

    def _compute(self, re_min, re_max, im_min, im_max):
        """
        Vectorized Mandelbrot set computation over a grid.
        Returns an integer array of iteration counts until divergence.
        """
        real = np.linspace(re_min, re_max, self.width)
        imag = np.linspace(im_min, im_max, self.height)
        X, Y = np.meshgrid(real, imag)
        C = X + 1j * Y
        Z = np.zeros_like(C)
        M = np.full(C.shape, self.max_iter, dtype=int)
        mask = np.ones(C.shape, bool)

        for i in range(self.max_iter):
            Z[mask] = Z[mask] * Z[mask] + C[mask]
            mask_new = np.abs(Z) <= 2
            diverged = mask & ~mask_new
            M[diverged] = i
            mask = mask_new

        return M

    def _draw(self):
        """Render the Mandelbrot set for the current view."""
        re_min, re_max, im_min, im_max = self.view
        self.fig.suptitle(os.path.basename(__file__), fontsize=14, fontweight="bold")
        with logger.timeit("Compute Mandelbrot"):
            data = self._compute(re_min, re_max, im_min, im_max)
        self.ax.clear()
        self.im = self.ax.imshow(
            data,
            extent=(re_min, re_max, im_min, im_max),
            cmap=self.cmap,
            origin="lower",
        )
        self.ax.set_title("Mandelbrot Set (drag to zoom, click to reset)")
        self.ax.set_xlabel("Re")
        self.ax.set_ylabel("Im")
        self.fig.canvas.draw()

    def _on_select(self, eclick, erelease):
        """Callback for rectangle selection: zoom or reset."""
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        if abs(x1 - x0) < 1e-6 and abs(y1 - y0) < 1e-6:
            self.view = self.default_view
        else:
            xmin, xmax = sorted([x0, x1])
            ymin, ymax = sorted([y0, y1])
            self.view = (xmin, xmax, ymin, ymax)
        self._draw()

    def _on_key(self, event):
        """Key press handler: 'r' to reset view."""
        if event.key in ("r", "R"):
            self.view = self.default_view
            self._draw()


def main():
    viewer = MandelbrotViewer()
    plt.show()


if __name__ == "__main__":
    main()
