import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from infrastructure.config import (
    WIDTH,
    HEIGHT,
    MAX_ITER,
    DEFAULT_VIEW,
    FIGURE_SIZE,
    COLORMAP,
)
import Metal
import MetalKit

# ---- CONFIGURABLE PARAMETERS ----
TARGET_RE = -0.743643887037158704752191506114774  # Example: Seahorse Valley
TARGET_IM = 0.131825904205311970493132056385139
ZOOM_STEPS = 120  # Number of frames in the animation
ZOOM_FACTOR = 0.97  # How much to shrink the window per frame (0.97 = 3% zoom per frame)

# ---- METAL MANDELBROT CLASS (from mandelbrot-metal.py) ----
METAL_SHADER = """
#include <metal_stdlib>
using namespace metal;

kernel void mandelbrot(device float* result [[buffer(0)]],
                      device atomic_uint* total_iterations [[buffer(1)]],
                      constant float& re_min [[buffer(2)]],
                      constant float& re_max [[buffer(3)]],
                      constant float& im_min [[buffer(4)]],
                      constant float& im_max [[buffer(5)]],
                      constant int& max_iter [[buffer(6)]],
                      constant int& width [[buffer(7)]],
                      constant int& height [[buffer(8)]],
                      uint2 position [[thread_position_in_grid]]) {
    float c_re = re_min + (re_max - re_min) * position.x / (width - 1);
    float c_im = im_min + (im_max - im_min) * position.y / (height - 1);
    float z_re = 0.0;
    float z_im = 0.0;
    int i = 0;
    while (z_re * z_re + z_im * z_im <= 4.0 && i < max_iter) {
        float z_re_sq = z_re * z_re;
        float z_im_sq = z_im * z_im;
        z_im = 2.0 * z_re * z_im + c_im;
        z_re = z_re_sq - z_im_sq + c_re;
        i++;
    }
    result[position.y * width + position.x] = i;
    atomic_fetch_add_explicit(total_iterations, i, memory_order_relaxed);
}
"""


class MandelbrotMetal:
    def __init__(self):
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if not self.device:
            raise RuntimeError("Metal is not supported on this device")
        self.command_queue = self.device.newCommandQueue()
        library = self.device.newLibraryWithSource_options_error_(
            METAL_SHADER, None, None
        )[0]
        self.mandelbrot_function = library.newFunctionWithName_("mandelbrot")
        self.pipeline = self.device.newComputePipelineStateWithFunction_error_(
            self.mandelbrot_function, None
        )[0]
        self.result_buffer = self.device.newBufferWithLength_options_(
            WIDTH * HEIGHT * 4, Metal.MTLResourceStorageModeShared
        )
        self.iterations_buffer = self.device.newBufferWithLength_options_(
            4, Metal.MTLResourceStorageModeShared
        )

    def compute(self, re_min, re_max, im_min, im_max):
        np.frombuffer(self.iterations_buffer.contents().as_buffer(4), dtype=np.uint32)[
            0
        ] = 0
        command_buffer = self.command_queue.commandBuffer()
        compute_encoder = command_buffer.computeCommandEncoder()
        compute_encoder.setComputePipelineState_(self.pipeline)
        compute_encoder.setBuffer_offset_atIndex_(self.result_buffer, 0, 0)
        compute_encoder.setBuffer_offset_atIndex_(self.iterations_buffer, 0, 1)
        compute_encoder.setBytes_length_atIndex_(
            np.array([re_min], dtype=np.float32).tobytes(), 4, 2
        )
        compute_encoder.setBytes_length_atIndex_(
            np.array([re_max], dtype=np.float32).tobytes(), 4, 3
        )
        compute_encoder.setBytes_length_atIndex_(
            np.array([im_min], dtype=np.float32).tobytes(), 4, 4
        )
        compute_encoder.setBytes_length_atIndex_(
            np.array([im_max], dtype=np.float32).tobytes(), 4, 5
        )
        compute_encoder.setBytes_length_atIndex_(
            np.array([MAX_ITER], dtype=np.int32).tobytes(), 4, 6
        )
        compute_encoder.setBytes_length_atIndex_(
            np.array([WIDTH], dtype=np.int32).tobytes(), 4, 7
        )
        compute_encoder.setBytes_length_atIndex_(
            np.array([HEIGHT], dtype=np.int32).tobytes(), 4, 8
        )
        grid_size = Metal.MTLSizeMake(WIDTH, HEIGHT, 1)
        thread_group_size = Metal.MTLSizeMake(16, 16, 1)
        compute_encoder.dispatchThreads_threadsPerThreadgroup_(
            grid_size, thread_group_size
        )
        compute_encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        result = np.frombuffer(
            self.result_buffer.contents().as_buffer(WIDTH * HEIGHT * 4),
            dtype=np.float32,
        ).reshape(HEIGHT, WIDTH)
        return result.astype(np.int32)


metal_compute = MandelbrotMetal()


def mandelbrot(re_min, re_max, im_min, im_max):
    return metal_compute.compute(re_min, re_max, im_min, im_max)


def interpolate_view(start, target, zoom_factor, step):
    re_min, re_max, im_min, im_max = start
    center_re = (re_min + re_max) / 2
    center_im = (im_min + im_max) / 2
    width0 = re_max - re_min
    height0 = im_max - im_min
    width = width0 * (zoom_factor**step)
    height = height0 * (zoom_factor**step)
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
ax.set_title("Mandelbrot Zoom Animation (Metal GPU)")
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


ani = FuncAnimation(fig, update, frames=ZOOM_STEPS, interval=20, blit=True, repeat=True)
plt.show()
