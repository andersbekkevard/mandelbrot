import time
import os
from contextlib import contextmanager
from datetime import datetime
import atexit
from collections import defaultdict


class MandelbrotLogger:
    LOGFILE = "mandelbrot.log"

    def __init__(self, filename=None):
        if filename is None:
            filename = os.path.basename(__file__)
        self.filename = os.path.basename(filename)
        self._print_header()
        self._timings = []
        atexit.register(self._write_logfile)

    def _print_header(self):
        bar = "=" * (len(self.filename) + 18)
        print(f"\n{bar}")
        print(f"  Mandelbrot Runner: {self.filename}  ")
        print(f"{bar}\n")

    @contextmanager
    def timeit(self, action_name):
        print(f"[⏳] {action_name}...", end="", flush=True)
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self._timings.append(elapsed)
        print(f" \033[92m✔\033[0m ({elapsed:.3f}s)")

    def _write_logfile(self):
        if self._timings:
            avg = sum(self._timings) / len(self._timings)
            dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"{dt} | {self.filename} | avg_compute_time={avg:.4f}s\n"
            with open(self.LOGFILE, "a") as f:
                f.write(line)


# Usage example in a mandelbrot file:
# from logger import MandelbrotLogger
# logger = MandelbrotLogger(__file__)
# with logger.timeit("Compute Mandelbrot"):
#     ...

if __name__ == "__main__":
    logfile = MandelbrotLogger.LOGFILE
    if not os.path.exists(logfile):
        print("No log file found.")
        exit(0)
    stats = defaultdict(lambda: {"count": 0, "total": 0.0})
    with open(logfile) as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue
            _, filename, avg_part = [p.strip() for p in parts]
            try:
                avg_time = float(avg_part.split("=")[-1][:-1])  # remove 's' at end
            except Exception:
                continue
            stats[filename]["count"] += 1
            stats[filename]["total"] += avg_time
    if not stats:
        print("No valid log entries found.")
        exit(0)
    # Compute averages
    results = []
    for fname, d in stats.items():
        avg = d["total"] / d["count"]
        results.append((fname, avg, d["count"]))
    # Sort by average time (ascending)
    results.sort(key=lambda x: x[1])
    # Print table
    print("\n================ Mandelbrot Benchmark Summary ================")
    print(f"{'Rank':<5} {'Script':<25} {'Avg Time (s)':<15} {'Runs':<5}")
    print("-------------------------------------------------------------")
    for i, (fname, avg, count) in enumerate(results, 1):
        print(f"{i:<5} {fname:<25} {avg:<15.4f} {count:<5}")
    print("=============================================================")
