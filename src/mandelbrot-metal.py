import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.backend_bases import MouseButton
from infrastructure.logger import MandelbrotLogger
import os
import Metal
import MetalKit
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

# Metal shader code
METAL_SHADER = """
#include <metal_stdlib>
using namespace metal;

kernel void mandelbrot(device float* result [[buffer(0)]],
                      constant float& re_min [[buffer(1)]],
                      constant float& re_max [[buffer(2)]],
                      constant float& im_min [[buffer(3)]],
                      constant float& im_max [[buffer(4)]],
                      constant int& max_iter [[buffer(5)]],
                      constant int& width [[buffer(6)]],
                      constant int& height [[buffer(7)]],
                      uint2 position [[thread_position_in_grid]]) {
    // Calculate complex number c
    float c_re = re_min + (re_max - re_min) * position.x / (width - 1);
    float c_im = im_min + (im_max - im_min) * position.y / (height - 1);
    
    // Initialize z
    float z_re = 0.0;
    float z_im = 0.0;
    int i = 0;
    
    // Compute Mandelbrot
    while (z_re * z_re + z_im * z_im <= 4.0 && i < max_iter) {
        float z_re_sq = z_re * z_re;
        float z_im_sq = z_im * z_im;
        z_im = 2.0 * z_re * z_im + c_im;
        z_re = z_re_sq - z_im_sq + c_re;
        i++;
    }
    
    // Store result
    result[position.y * width + position.x] = i;
}
"""


class MandelbrotMetal:
    def __init__(self):
        # Initialize Metal
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if not self.device:
            raise RuntimeError("Metal is not supported on this device")

        # Create command queue
        self.command_queue = self.device.newCommandQueue()

        # Create library and function
        library = self.device.newLibraryWithSource_options_error_(
            METAL_SHADER, None, None
        )[0]
        self.mandelbrot_function = library.newFunctionWithName_("mandelbrot")

        # Create compute pipeline
        self.pipeline = self.device.newComputePipelineStateWithFunction_error_(
            self.mandelbrot_function, None
        )[0]

        # Create buffers
        self.result_buffer = self.device.newBufferWithLength_options_(
            WIDTH * HEIGHT * 4, Metal.MTLResourceStorageModeShared
        )

    def compute(self, re_min, re_max, im_min, im_max):
        # Create command buffer
        command_buffer = self.command_queue.commandBuffer()
        compute_encoder = command_buffer.computeCommandEncoder()

        # Set pipeline and function
        compute_encoder.setComputePipelineState_(self.pipeline)

        # Set buffers
        compute_encoder.setBuffer_offset_atIndex_(self.result_buffer, 0, 0)
        compute_encoder.setBytes_length_atIndex_(
            np.array([re_min], dtype=np.float32).tobytes(), 4, 1
        )
        compute_encoder.setBytes_length_atIndex_(
            np.array([re_max], dtype=np.float32).tobytes(), 4, 2
        )
        compute_encoder.setBytes_length_atIndex_(
            np.array([im_min], dtype=np.float32).tobytes(), 4, 3
        )
        compute_encoder.setBytes_length_atIndex_(
            np.array([im_max], dtype=np.float32).tobytes(), 4, 4
        )
        compute_encoder.setBytes_length_atIndex_(
            np.array([MAX_ITER], dtype=np.int32).tobytes(), 4, 5
        )
        compute_encoder.setBytes_length_atIndex_(
            np.array([WIDTH], dtype=np.int32).tobytes(), 4, 6
        )
        compute_encoder.setBytes_length_atIndex_(
            np.array([HEIGHT], dtype=np.int32).tobytes(), 4, 7
        )

        # Set grid size
        grid_size = Metal.MTLSizeMake(WIDTH, HEIGHT, 1)
        thread_group_size = Metal.MTLSizeMake(16, 16, 1)

        # Dispatch compute command
        compute_encoder.dispatchThreads_threadsPerThreadgroup_(
            grid_size, thread_group_size
        )
        compute_encoder.endEncoding()

        # Execute command buffer
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        # Get results
        result = np.frombuffer(
            self.result_buffer.contents().as_buffer(WIDTH * HEIGHT * 4),
            dtype=np.float32,
        ).reshape(HEIGHT, WIDTH)

        return result.astype(np.int32)


# Initialize Metal compute
metal_compute = MandelbrotMetal()


def mandelbrot(re_min, re_max, im_min, im_max):
    return metal_compute.compute(re_min, re_max, im_min, im_max)


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
