import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

width, height = 800, 600
# Plot window
re_min, re_max = -2.0, 1.0
im_min, im_max = -1.0, 1.0
# Maximum number of iterations
max_iter = 100

# Store default view
default_view = (re_min, re_max, im_min, im_max)


def compute_mandelbrot(re_min, re_max, im_min, im_max, width, height, max_iter):
    mandelbrot_set = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            c = complex(
                re_min + (x / width) * (re_max - re_min),
                im_min + (y / height) * (im_max - im_min),
            )
            z = 0
            iter = 0
            while abs(z) <= 2 and iter < max_iter:
                z = z * z + c
                iter += 1
            mandelbrot_set[y, x] = iter
    return mandelbrot_set


# Initial view
current_view = (re_min, re_max, im_min, im_max)

fig, ax = plt.subplots(figsize=(10, 7.5))
mandelbrot_img = None
rect_patch = None
press_event = None


# Draw Mandelbrot set
def draw_mandelbrot():
    global mandelbrot_img
    re_min, re_max, im_min, im_max = current_view
    mandelbrot_set = compute_mandelbrot(
        re_min, re_max, im_min, im_max, width, height, max_iter
    )
    ax.clear()
    mandelbrot_img = ax.imshow(
        mandelbrot_set,
        extent=(re_min, re_max, im_min, im_max),
        cmap="hot",
        origin="lower",
    )
    ax.set_title("Mandelbrot Set (Drag to zoom, click to reset)")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    plt.draw()


def remove_rect_patch():
    global rect_patch
    if rect_patch is not None:
        rect_patch.set_visible(False)
    rect_patch = None


def on_press(event):
    global press_event, rect_patch
    if event.inaxes != ax:
        return
    press_event = event
    # Remove old rectangle if exists
    remove_rect_patch()


def on_release(event):
    global press_event, current_view, rect_patch
    if event.inaxes != ax or press_event is None:
        return
    # If mouse was not dragged, treat as single click (reset)
    if (
        abs(event.xdata - press_event.xdata) < 1e-5
        and abs(event.ydata - press_event.ydata) < 1e-5
    ):
        current_view = default_view
        draw_mandelbrot()
        press_event = None
        return
    # Get rectangle bounds
    x0, y0 = press_event.xdata, press_event.ydata
    x1, y1 = event.xdata, event.ydata
    xmin, xmax = sorted([x0, x1])
    ymin, ymax = sorted([y0, y1])
    # Update view
    current_view = (xmin, xmax, ymin, ymax)
    draw_mandelbrot()
    press_event = None
    # Remove rectangle
    remove_rect_patch()


def on_motion(event):
    global rect_patch
    if press_event is None or event.inaxes != ax:
        return
    x0, y0 = press_event.xdata, press_event.ydata
    x1, y1 = event.xdata, event.ydata
    xmin, xmax = min(x0, x1), max(x0, x1)
    ymin, ymax = min(y0, y1), max(y0, y1)
    # Remove old rectangle
    remove_rect_patch()
    # Draw new rectangle
    rect_patch = Rectangle(
        (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color="cyan", linewidth=2
    )
    ax.add_patch(rect_patch)
    plt.draw()


fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("motion_notify_event", on_motion)

draw_mandelbrot()
plt.show()
