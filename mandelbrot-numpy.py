import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.backend_bases import MouseButton
from logger import MandelbrotLogger
import os

# Parameters
W, H, MAX_ITER = 800, 600, 100  # Standardized for benchmarking
DEFAULT_VIEW = (-2.0, 1.0, -1.0, 1.0)

view = list(DEFAULT_VIEW)
fig, ax = plt.subplots(figsize=(8, 6))

logger = MandelbrotLogger(__file__)


# Mandelbrot calculation (pure numpy)
def mandelbrot(re_min, re_max, im_min, im_max):
    x = np.linspace(re_min, re_max, W)
    y = np.linspace(im_min, im_max, H)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C)
    M = np.full(C.shape, MAX_ITER, dtype=int)
    mask = np.ones(C.shape, bool)
    for i in range(MAX_ITER):
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        mask_new = np.abs(Z) <= 2
        M[mask & ~mask_new] = i
        mask = mask_new
    return M


def draw():
    ax.clear()
    fig.suptitle(os.path.basename(__file__), fontsize=14, fontweight="bold")
    with logger.timeit("Compute Mandelbrot"):
        m = mandelbrot(*view)
    ax.imshow(
        m, extent=(view[0], view[1], view[2], view[3]), cmap="hot", origin="lower"
    )
    ax.set_title("Mandelbrot Set (drag to zoom, click to reset)")
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


selector = RectangleSelector(
    ax,
    on_select,
    useblit=True,
    button=[MouseButton.LEFT],
    spancoords="data",
    interactive=False,
)
draw()
plt.show()
