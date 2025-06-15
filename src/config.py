"""
Central configuration for all Mandelbrot implementations.
Adjust these parameters to control the computational load and visualization.
"""

# Configuration presets
PRESETS = {
    "easy": {
        "WIDTH": 800,
        "HEIGHT": 600,
        "MAX_ITER": 100,
        "FIGURE_SIZE": (8, 6),
    },
    "medium": {
        "WIDTH": 1200,
        "HEIGHT": 900,
        "MAX_ITER": 150,
        "FIGURE_SIZE": (8, 6),
    },
    "hard": {
        "WIDTH": 2000,
        "HEIGHT": 1500,
        "MAX_ITER": 2000,
        "FIGURE_SIZE": (8, 6),
    },
}

# Current preset selection
CURRENT_PRESET = "medium"  # Change this to switch between presets

# Default view coordinates (re_min, re_max, im_min, im_max)
DEFAULT_VIEW = (-2.0, 1.0, -1.0, 1.0)

# Visualization parameters
COLORMAP = "hot"

# Export the current preset's parameters
preset = PRESETS[CURRENT_PRESET]
WIDTH = preset["WIDTH"]
HEIGHT = preset["HEIGHT"]
MAX_ITER = preset["MAX_ITER"]
FIGURE_SIZE = preset["FIGURE_SIZE"]
