import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.backend_bases import MouseButton
from mandelbrot_rust import compute_mandelbrot
from logger import MandelbrotLogger
import os
from config import WIDTH, HEIGHT, MAX_ITER, DEFAULT_VIEW, FIGURE_SIZE, COLORMAP

# Initialize view and figure
view = list(DEFAULT_VIEW)
fig, ax = plt.subplots(figsize=FIGURE_SIZE)
logger = MandelbrotLogger(os.path.basename(__file__))


def draw():
    """Draw the Mandelbrot set with current view parameters."""
    ax.clear()
    fig.suptitle(os.path.basename(__file__), fontsize=14, fontweight="bold")

    with logger.timeit("Compute Mandelbrot"):
        result = compute_mandelbrot(WIDTH, HEIGHT, MAX_ITER, *view)

    ax.imshow(
        result,
        extent=(view[0], view[1], view[2], view[3]),
        cmap=COLORMAP,
        origin="lower",
    )
    ax.set_title(
        f"Mandelbrot Set (drag to zoom, click to reset)\nResolution: {WIDTH}x{HEIGHT}, Max Iterations: {MAX_ITER}"
    )
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    plt.draw()


def on_select(eclick, erelease):
    """Handle rectangle selection for zooming."""
    x0, y0 = eclick.xdata, eclick.ydata
    x1, y1 = erelease.xdata, erelease.ydata

    if abs(x1 - x0) < 1e-6 and abs(y1 - y0) < 1e-6:
        # Reset to default view if selection is too small
        view[:] = DEFAULT_VIEW
    else:
        # Update view with new boundaries
        view[:] = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
    draw()


# Initial draw
draw()

# Create rectangle selector for zooming
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
