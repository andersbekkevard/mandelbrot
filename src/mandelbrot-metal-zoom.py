import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

# =============================================================================
# ZOOM CONFIGURATION - Edit these to customize the animation
# =============================================================================

# Predefined zoom locations - incredibly precise coordinates of famous regions
ZOOM_LOCATIONS = {
    # Classic Seahorse Valley - the most famous Mandelbrot location
    "Seahorse Valley": (-0.75006, -0.74994, 0.09995, 0.10005),
    # Lightning Lake - dramatic tendrils and filaments
    "Lightning Lake": (-1.77501, -1.77499, -0.00001, 0.00001),
    # Double Spiral Valley - intricate dual spiral patterns
    "Double Spiral Valley": (-0.74843, -0.74837, 0.11307, 0.11313),
    # Mini Mandelbrot - self-similar copy of the full set
    "Mini Mandelbrot": (-0.7269, -0.7263, 0.1103, 0.1109),
    # Scepter Valley - between period-2 and period-4 bulbs
    "Scepter Valley": (-1.25, -1.24999, 0.0, 0.00001),
    # Feather Valley - delicate feather-like structures
    "Feather Valley": (-0.23512, -0.23508, 0.82703, 0.82707),
    # Dragon Valley - serpentine spirals and tendrils
    "Dragon Valley": (-0.16070135, -0.16070125, 1.0375665, 1.0375675),
    # Burning Ship Valley - ship-like structures (burning ship fractal style)
    "Burning Ship Valley": (-1.7625, -1.7615, -0.0036, -0.0026),
    # Misiurewicz Point - famous mathematical branch point
    "Misiurewicz Point": (-1.5437, -1.5436, 0.0, 0.0001),
    # Spiral Beach - beautiful spiral formations
    "Spiral Beach": (-0.16, -0.159, 1.0251, 1.0261),
    # Tendril Forest - dense network of hair-like tendrils
    "Tendril Forest": (-0.99, -0.989, 0.275, 0.276),
    # Default full view
    "Default View": DEFAULT_VIEW,
}

# Select zoom target here - change this line to switch locations!
CURRENT_LOCATION = "Burning Ship Valley"

# Animation settings
ZOOM_SPEED = 0.99  # Zoom factor per frame (smaller = faster zoom)
ANIMATION_FPS = 60  # Frames per second
MAX_ZOOM_FRAMES = 1000  # Maximum number of zoom frames
LOOP_ANIMATION = True  # Whether to loop the animation

# =============================================================================
# METAL INFRASTRUCTURE (from mandelbrot-metal.py)
# =============================================================================

# Metal shader code
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
    
    // Add to total iterations (atomic operation)
    atomic_fetch_add_explicit(total_iterations, i, memory_order_relaxed);
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
        self.iterations_buffer = self.device.newBufferWithLength_options_(
            4, Metal.MTLResourceStorageModeShared
        )

    def compute(self, re_min, re_max, im_min, im_max):
        # Reset iterations counter
        np.frombuffer(self.iterations_buffer.contents().as_buffer(4), dtype=np.uint32)[
            0
        ] = 0

        # Create command buffer
        command_buffer = self.command_queue.commandBuffer()
        compute_encoder = command_buffer.computeCommandEncoder()

        # Set pipeline and function
        compute_encoder.setComputePipelineState_(self.pipeline)

        # Set buffers
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

        # Get total iterations
        total_iterations = np.frombuffer(
            self.iterations_buffer.contents().as_buffer(4), dtype=np.uint32
        )[0]

        return result.astype(np.int32), total_iterations


# =============================================================================
# ZOOM ANIMATION LOGIC
# =============================================================================


class MandelbrotZoomAnimation:
    def __init__(self):
        # Initialize Metal compute
        self.metal_compute = MandelbrotMetal()
        self.logger = MandelbrotLogger(__file__)

        # Get target location
        if CURRENT_LOCATION not in ZOOM_LOCATIONS:
            raise ValueError(
                f"Unknown location: {CURRENT_LOCATION}. Available: {list(ZOOM_LOCATIONS.keys())}"
            )

        self.target_location = ZOOM_LOCATIONS[CURRENT_LOCATION]
        self.start_location = DEFAULT_VIEW

        # Animation state
        self.frame = 0
        self.current_view = list(self.start_location)

        # Setup matplotlib
        self.fig, self.ax = plt.subplots(figsize=FIGURE_SIZE)
        self.fig.suptitle(
            f"Mandelbrot Metal Zoom Animation", fontsize=14, fontweight="bold"
        )
        self.im = None

        print(f"\n🎯 Zoom Animation Settings:")
        print(f"   Target: {CURRENT_LOCATION}")
        print(f"   Coordinates: {self.target_location}")
        print(f"   Frames: {MAX_ZOOM_FRAMES}")
        print(f"   FPS: {ANIMATION_FPS}")
        print(f"   Zoom Speed: {ZOOM_SPEED}")

    def calculate_zoom_level(self, frame):
        """Calculate smooth zoom interpolation"""
        if frame >= MAX_ZOOM_FRAMES:
            return 1.0

        # Exponential zoom progression
        progress = frame / MAX_ZOOM_FRAMES
        zoom_factor = ZOOM_SPEED**frame
        return zoom_factor

    def interpolate_view(self, zoom_factor):
        """Interpolate between start and target view"""
        start = np.array(self.start_location)
        target = np.array(self.target_location)

        # Calculate center points
        start_center = [(start[0] + start[1]) / 2, (start[2] + start[3]) / 2]
        target_center = [(target[0] + target[1]) / 2, (target[2] + target[3]) / 2]

        # Interpolate center
        current_center = [
            start_center[0] + (target_center[0] - start_center[0]) * (1 - zoom_factor),
            start_center[1] + (target_center[1] - start_center[1]) * (1 - zoom_factor),
        ]

        # Calculate size based on zoom
        start_size = [(start[1] - start[0]) / 2, (start[3] - start[2]) / 2]
        size = [start_size[0] * zoom_factor, start_size[1] * zoom_factor]

        # Create new view
        return [
            current_center[0] - size[0],  # re_min
            current_center[0] + size[0],  # re_max
            current_center[1] - size[1],  # im_min
            current_center[1] + size[1],  # im_max
        ]

    def animate_frame(self, frame_num):
        """Animation function called by matplotlib"""
        # Handle looping
        if LOOP_ANIMATION and frame_num >= MAX_ZOOM_FRAMES:
            frame_num = frame_num % MAX_ZOOM_FRAMES

        self.frame = frame_num

        # Calculate current zoom level
        zoom_factor = self.calculate_zoom_level(frame_num)
        self.current_view = self.interpolate_view(zoom_factor)

        # Compute Mandelbrot with timing
        with self.logger.timeit("Metal GPU Compute"):
            mandelbrot_data, total_iterations = self.metal_compute.compute(
                *self.current_view
            )

        # Update display
        self.ax.clear()

        # Format iterations
        if total_iterations >= 1e9:
            iterations_str = f"{total_iterations/1e9:.1f}B"
        elif total_iterations >= 1e6:
            iterations_str = f"{total_iterations/1e6:.1f}M"
        elif total_iterations >= 1e3:
            iterations_str = f"{total_iterations/1e3:.1f}K"
        else:
            iterations_str = str(total_iterations)

        # Calculate zoom level for display
        zoom_level = 1.0 / zoom_factor if zoom_factor > 0 else float("inf")

        # Display image
        self.ax.imshow(
            mandelbrot_data, extent=self.current_view, cmap=COLORMAP, origin="lower"
        )

        self.ax.set_title(
            f"Location: {CURRENT_LOCATION} | Frame: {frame_num+1}/{MAX_ZOOM_FRAMES}\n"
            f"Zoom: {zoom_level:.2e}x | Iterations: {iterations_str} | Resolution: {WIDTH}x{HEIGHT}",
            fontsize=10,
        )

        self.ax.set_xlabel("Real")
        self.ax.set_ylabel("Imaginary")

        return [self.ax]

    def start_animation(self):
        """Start the zoom animation"""
        interval = 1000 / ANIMATION_FPS  # milliseconds per frame

        # Create animation
        if LOOP_ANIMATION:
            frames = None  # Infinite loop
        else:
            frames = MAX_ZOOM_FRAMES

        anim = animation.FuncAnimation(
            self.fig,
            self.animate_frame,
            frames=frames,
            interval=interval,
            blit=False,
            repeat=LOOP_ANIMATION,
        )

        plt.tight_layout()
        plt.show()

        return anim


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    try:
        # Create and start animation
        zoom_anim = MandelbrotZoomAnimation()
        animation_obj = zoom_anim.start_animation()

    except KeyboardInterrupt:
        print("\n= Animation stopped by user")
    except Exception as e:
        print(f"\nL Error: {e}")
        print("\n= Make sure you have:")
        print("   - macOS with Metal support")
        print("   - metalgpu package installed: pip install metalgpu")
        print("   - All required dependencies from requirements.txt")
