import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.backend_bases import MouseButton
from infrastructure.logger import MandelbrotLogger
import os
import torch
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

# Check for available devices (MPS for Apple Silicon, CUDA for NVIDIA, CPU as fallback)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


@torch.jit.script
def mandelbrot_kernel(
    c_re: torch.Tensor, c_im: torch.Tensor, max_iter: int
) -> torch.Tensor:
    # Initialize z and result tensors
    z_re = torch.zeros_like(c_re)
    z_im = torch.zeros_like(c_im)
    result = torch.full_like(c_re, max_iter, dtype=torch.int32)

    # Create mask for points that haven't escaped
    mask = torch.ones_like(c_re, dtype=torch.bool)

    # Compute Mandelbrot set
    for i in range(max_iter):
        # Only compute for points that haven't escaped
        z_re_sq = z_re[mask] * z_re[mask]
        z_im_sq = z_im[mask] * z_im[mask]

        # Update z values
        z_im_new = 2.0 * z_re[mask] * z_im[mask] + c_im[mask]
        z_re_new = z_re_sq - z_im_sq + c_re[mask]

        # Update mask for points that have escaped
        escaped = (z_re_sq + z_im_sq) > 4.0

        # Update values
        z_re[mask] = z_re_new
        z_im[mask] = z_im_new
        result[mask] = torch.where(escaped, i, result[mask])

        # Update mask
        new_mask = mask.clone()
        new_mask[mask] = ~escaped
        mask = new_mask

        # Early exit if all points have escaped
        if not mask.any():
            break

    return result


def mandelbrot(re_min, re_max, im_min, im_max):
    # Create the complex plane using separate real and imaginary parts
    re = torch.linspace(re_min, re_max, WIDTH, device=device)
    im = torch.linspace(im_min, im_max, HEIGHT, device=device)

    # Create the complex grid using broadcasting
    c_re = re.unsqueeze(0).expand(HEIGHT, -1)
    c_im = im.unsqueeze(1).expand(-1, WIDTH)

    # Compute Mandelbrot set using the JIT-compiled kernel
    result = mandelbrot_kernel(c_re, c_im, MAX_ITER)

    # Move result to CPU and convert to numpy
    return result.cpu().numpy()


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
