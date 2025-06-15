import sys
import os
import importlib.util
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent.parent / "src"
sys.path.append(str(src_dir))
sys.path.append(str(src_dir / "infrastructure"))

import matplotlib

matplotlib.use("Agg")  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt

# Import the logger
from infrastructure.logger import MandelbrotLogger
from infrastructure.config import DEFAULT_VIEW

# List of implementations to test
IMPLEMENTATIONS = [
    # "mandelbrot-vanilla.py",
    # "mandelbrot-numpy.py",
    # "mandelbrot-parallel.py",
    "mandelbrot-numba.py",
    "mandelbrot-numba-parallel.py",
    "mandelbrot-rust.py",
    # "mandelbrot-codex.py"
]

# Define zoom regions to test
ZOOM_REGIONS = [
    DEFAULT_VIEW,  # Default view
    [-0.5, 0.5, -0.5, 0.5],  # Center zoom
    [-0.1, 0.1, -0.1, 0.1],  # Deep zoom
]


def load_module(module_path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location("module", module_path)
    if spec is None:
        raise ImportError(f"Could not load module from {module_path}")
    if spec.loader is None:
        raise ImportError(f"No loader found for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_implementation(impl_path, zoom_regions):
    """Test a single implementation with multiple zoom regions."""
    print(f"\nTesting {impl_path}...")

    try:
        # Load the implementation
        module = load_module(os.path.join(src_dir, impl_path))

        # Test each zoom region
        for i, region in enumerate(zoom_regions):
            print(f"  Zoom level {i+1}: {region}")
            module.view[:] = region
            module.draw()
            plt.close("all")  # Close all figures after each draw
    except Exception as e:
        print(f"  Error testing {impl_path}: {str(e)}")
        plt.close("all")


def main():
    print("Starting performance test of all Mandelbrot implementations...")
    print(
        f"Testing {len(IMPLEMENTATIONS)} implementations with {len(ZOOM_REGIONS)} zoom levels each"
    )

    for impl in IMPLEMENTATIONS:
        test_implementation(impl, ZOOM_REGIONS)

    print("\nTest completed. Check mandelbrot.log for performance data.")


if __name__ == "__main__":
    main()
