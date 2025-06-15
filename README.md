# Mandelbrot Set Computation Benchmark

This project implements and benchmarks various approaches to computing the Mandelbrot set, exploring different programming languages and optimization techniques.

## Performance Summary

Latest benchmark results (average computation time in seconds):

| Rank | Implementation          | Easy   | Medium | Hard   |
|------|------------------------|--------|--------|--------|
| 1    | Metal GPU              | 0.003  | 0.004  | 0.022  |
| 2    | Rust                   | 0.013  | 0.046  | 1.371  |
| 3    | Numba                  | 0.055  | 0.173  | 5.011  |
| 4    | Numba Parallel         | 0.072  | 0.088  | 0.940  |
| 5    | NumPy                  | 0.194  | 0.745  | -      |
| 6    | Codex                  | 0.294  | 0.759  | -      |
| 7    | Python Parallel        | 0.594  | 1.048  | -      |
| 8    | Pure Python            | 0.898  | 4.600  | -      |
| 9    | PyTorch GPU            | -      | 4.909  | 30.098 |

Metal GPU is consistently fastest across all modes, with Rust and Numba providing the best CPU-based alternatives. PyTorch GPU shows significantly higher computation times compared to Metal.

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
