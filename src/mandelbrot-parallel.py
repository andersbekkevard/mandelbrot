import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.backend_bases import MouseButton
from infrastructure.logger import MandelbrotLogger
import os
from multiprocessing import Pool, cpu_count
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


def compute_chunk(args):
    """Compute a chunk of the Mandelbrot set."""
    start_y, end_y, re_min, re_max, im_min, im_max = args
    result = []

    for y in range(start_y, end_y):
        row = []
        im = im_min + (im_max - im_min) * y / (HEIGHT - 1)
        for x in range(WIDTH):
            re = re_min + (re_max - re_min) * x / (WIDTH - 1)
            c = complex(re, im)
            z = 0
            count = 0
            while abs(z) <= 2 and count < MAX_ITER:
                z = z * z + c
                count += 1
            row.append(count)
        result.append(row)

    return result


def mandelbrot(re_min, re_max, im_min, im_max):
    """Compute the Mandelbrot set using parallel processing."""
    # Create a pool of workers
    n_processes = max(1, cpu_count() - 1)  # Leave one core free
    with Pool(processes=n_processes) as pool:
        # Split the work into chunks
        chunk_size = HEIGHT // n_processes
        chunks = []
        for i in range(0, HEIGHT, chunk_size):
            end_y = min(i + chunk_size, HEIGHT)
            chunks.append((i, end_y, re_min, re_max, im_min, im_max))

        # Compute chunks in parallel
        results = pool.map(compute_chunk, chunks)

    # Combine results
    return np.array([row for chunk in results for row in chunk])


def main():
    logger = MandelbrotLogger(__file__)

    def timed_draw():
        with logger.timeit("Compute Mandelbrot"):
            m = mandelbrot(*view)
        ax.clear()
        fig.suptitle(os.path.basename(__file__), fontsize=14, fontweight="bold")
        ax.imshow(
            m,
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
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        if abs(x1 - x0) < 1e-6 and abs(y1 - y0) < 1e-6:
            view[:] = DEFAULT_VIEW
        else:
            view[:] = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
        timed_draw()

    timed_draw()
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
