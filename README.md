# Mandelbrot Set Computation Benchmark

This project implements and benchmarks various approaches to computing the Mandelbrot set, exploring different programming languages and optimization techniques.

## Performance Summary

Latest benchmark results (average computation time in seconds):

| Rank | Implementation    | Easy   | Medium | Hard    |
|------|------------------|--------|--------|---------|
| 1    | Metal (GPU)      | 0.0029 | 0.0042 | 0.0234  |
| 2    | Rust             | 0.0138 | 0.0475 | 1.3709  |
| 3    | Numba Parallel   | 0.0638 | 0.0847 | 0.9253  |
| 4    | Numba            | 0.0779 | 0.1822 | 5.0108  |
| 5    | NumPy            | 0.2086 | 0.7475 | 30.4738 |
| 6    | Codex (OOP-impl) | 0.2886 | 0.7662 | 28.2540 |
| 7    | Python Parallel  | 0.5938 | 1.0478 | 25.0336 |
| 8    | Torch (GPU)      | 1.7891 | 4.5363 | 62.1589 |
| 9    | Pure Python      | 1.1413 | 4.7807 | 93.8319 |

- Lower is better. All times in seconds. All implementations tested on all presets.

Metal (GPU) is consistently fastest, followed by Rust and Numba (parallel). Pure Python and Torch are the slowest, especially for harder presets.

## Implementations

The project includes several implementations of the Mandelbrot set computation:

1. **Python Implementations**:
   - `mandelbrot-vanilla.py`: Pure Python implementation
   - `mandelbrot-numpy.py`: Using NumPy for vectorized computation
   - `mandelbrot-numba.py`: JIT-compiled Python using Numba
   - `mandelbrot-numba-parallel.py`: Parallel Numba implementation
   - `mandelbrot-parallel.py`: Python multiprocessing implementation
   - `mandelbrot-torch.py`: PyTorch GPU implementation
   - `mandelbrot-metal.py`: Apple Metal GPU implementation

2. **Rust Implementation**:
   - `mandelbrot_rust/`: Parallel Rust implementation using rayon
   - Exposed as a Python module for easy benchmarking

## Performance Tracking

The project includes a logging system that tracks computation times for each implementation across different presets:
- Easy: Basic computation parameters
- Medium: More complex parameters
- Hard: Most demanding parameters

Results are logged to `mandelbrot.log` and can be viewed using the `print_stats()` function in `src/infrastructure/logger.py`.

## Requirements

- Python 3.x
- NumPy
- Numba
- PyTorch (for GPU implementation)
- Rust (for Rust implementation)
- Apple Metal (for Metal implementation)

## Usage

Run any implementation directly:
```bash
python src/mandelbrot-numba.py
```

View performance statistics:
```bash
python src/infrastructure/logger.py
```

## Project Structure

```
.
├── src/
│   ├── infrastructure/
│   │   ├── logger.py
│   │   └── config.py
│   ├── mandelbrot_rust/
│   │   └── src/
│   │       └── lib.rs
│   └── mandelbrot-*.py
├── tests/
└── mandelbrot.log
```

## Performance Considerations

The different implementations demonstrate various optimization techniques:
- Vectorization (NumPy)
- JIT compilation (Numba)
- Parallel processing (multiprocessing, rayon)
- GPU acceleration (PyTorch, Metal)
- Low-level optimization (Rust)

Each implementation offers different trade-offs between:
- Development complexity
- Performance
- Hardware requirements
- Maintainability
