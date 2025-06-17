import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import jit, prange
from infrastructure.config import (
    WIDTH,
    HEIGHT,
    MAX_ITER,
    DEFAULT_VIEW,
    FIGURE_SIZE,
    COLORMAP,
)

# ---- CONFIGURABLE PARAMETERS ----
TARGET_RE = -0.743643887037158704752191506114774  # Example: Seahorse Valley
TARGET_IM = 0.131825904205311970493132056385139
ZOOM_STEPS = 120  # Number of frames in the animation
ZOOM_FACTOR = 0.97  # How much to shrink the window per frame (0.97 = 3% zoom per frame)


@jit(nopython=True, parallel=True, fastmath=True)
def mandelbrot_kernel(re_min, re_max, im_min, im_max):
    result = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    for y in prange(HEIGHT):
        for x in range(WIDTH):
            c_re = re_min + (re_max - re_min) * x / (WIDTH - 1)
            c_im = im_min + (im_max - im_min) * y / (HEIGHT - 1)
            z_re = 0.0
            z_im = 0.0
            i = 0
            while z_re * z_re + z_im * z_im <= 4.0 and i < MAX_ITER:
                z_re_sq = z_re * z_re
                z_im_sq = z_im * z_im
                z_im = 2.0 * z_re * z_im + c_im
                z_re = z_re_sq - z_im_sq + c_re
                i += 1
            result[y, x] = i
    return result


def mandelbrot(re_min, re_max, im_min, im_max):
    return mandelbrot_kernel(re_min, re_max, im_min, im_max)


def interpolate_view(start, target, zoom_factor, step):
    # Always start with the full DEFAULT_VIEW, then zoom toward the target
    re_min, re_max, im_min, im_max = start
    center_re = (re_min + re_max) / 2
    center_im = (im_min + im_max) / 2
    width0 = re_max - re_min
    height0 = im_max - im_min
    # The zoom factor shrinks the window each step
    width = width0 * (zoom_factor**step)
    height = height0 * (zoom_factor**step)
    # Interpolate center toward the target
    interp = 1 - (zoom_factor**step)
    new_center_re = center_re * (1 - interp) + target[0] * interp
    new_center_im = center_im * (1 - interp) + target[1] * interp
    new_re_min = new_center_re - width / 2
    new_re_max = new_center_re + width / 2
    new_im_min = new_center_im - height / 2
    new_im_max = new_center_im + height / 2
    return new_re_min, new_re_max, new_im_min, new_im_max


fig, ax = plt.subplots(figsize=FIGURE_SIZE)
img = ax.imshow(
    np.zeros((HEIGHT, WIDTH)),
    extent=DEFAULT_VIEW,
    cmap=COLORMAP,
    origin="lower",
    vmin=0,
    vmax=MAX_ITER,
)
ax.set_title("Mandelbrot Zoom Animation (Numba Parallel)")
ax.set_xlabel("Re")
ax.set_ylabel("Im")


def update(frame):
    view = interpolate_view(DEFAULT_VIEW, (TARGET_RE, TARGET_IM), ZOOM_FACTOR, frame)
    m = mandelbrot(*view)
    if frame == 0:
        print(f"Initial view: {view}")
        print(f"Data min/max: {m.min()} / {m.max()}")
    img.set_data(m)
    img.set_extent((view[0], view[1], view[2], view[3]))
    ax.set_title(f"Center: ({(view[0]+view[1])/2:.6f}, {(view[2]+view[3])/2:.6f})")
    return [img]


ani = FuncAnimation(fig, update, frames=ZOOM_STEPS, interval=50, blit=True, repeat=True)
plt.show()
