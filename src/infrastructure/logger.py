import time
import logging
import os
from contextlib import contextmanager
from datetime import datetime
import atexit
from collections import defaultdict
import sys
import os

# Add the infrastructure directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CURRENT_PRESET


class MandelbrotLogger:
    LOGFILE = "mandelbrot.log"

    def __init__(self, filename=None):
        if filename is None:
            filename = os.path.basename(__file__)
        self.filename = os.path.basename(filename)
        self._print_header()
        self._timings = []
        # Ensure the log file exists
        if not os.path.exists(self.LOGFILE):
            with open(self.LOGFILE, "w") as f:
                f.write("")  # Create empty file
        atexit.register(self._write_logfile)

    def _print_header(self):
        bar = "=" * (len(self.filename) + 18)
        print(f"\n{bar}")
        print(f"  Mandelbrot Runner: {self.filename}  ")
        print(f"  Config Preset: {CURRENT_PRESET}  ")
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
            try:
                avg = sum(self._timings) / len(self._timings)
                dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                line = f"{dt} | {self.filename} | {CURRENT_PRESET} | avg_compute_time={avg:.4f}s\n"

                # Debug print
                print(f"\nWriting to log file: {line.strip()}")

                with open(self.LOGFILE, "a") as f:
                    f.write(line)
                    f.flush()  # Ensure the write is completed
                    os.fsync(f.fileno())  # Force write to disk
            except Exception as e:
                print(f"\nError writing to log file: {e}")


def print_stats():
    """Print statistics for each preset separately, in the order: easy, medium, hard."""
    logfile = MandelbrotLogger.LOGFILE
    if not os.path.exists(logfile):
        print("No log file found.")
        return

    # Group stats by preset
    stats = defaultdict(lambda: defaultdict(lambda: {"count": 0, "total": 0.0}))

    with open(logfile) as f:
        for line in f:
            parts = line.strip().split("|")
            if (
                len(parts) != 4
            ):  # Now expecting 4 parts: timestamp, filename, preset, time
                continue
            _, filename, preset, avg_part = [p.strip() for p in parts]
            try:
                avg_time = float(avg_part.split("=")[-1][:-1])  # remove 's' at end
            except Exception:
                continue
            stats[preset][filename]["count"] += 1
            stats[preset][filename]["total"] += avg_time

    if not stats:
        print("No valid log entries found.")
        return

    # Print stats for each preset in the desired order
    preset_order = ["easy", "medium", "hard"]
    col_script = 30
    col_avg = 18
    col_runs = 7
    for preset in preset_order:
        if preset not in stats:
            continue
        preset_stats = stats[preset]
        print(
            f"\n================ Mandelbrot Benchmark Summary ({preset}) ================"
        )
        print(
            f"{'Rank':<5} {'Script':<{col_script}} {'Avg Time (s)':<{col_avg}} {'Runs':<{col_runs}}"
        )
        print("-" * (5 + 1 + col_script + 1 + col_avg + 1 + col_runs))

        # Compute averages for this preset
        results = []
        for fname, d in preset_stats.items():
            avg = d["total"] / d["count"]
            results.append((fname, avg, d["count"]))

        # Sort by average time (ascending)
        results.sort(key=lambda x: x[1])

        # Print results
        for i, (fname, avg, count) in enumerate(results, 1):
            print(
                f"{i:<5} {fname:<{col_script}} {avg:<{col_avg}.4f} {count:<{col_runs}}"
            )
        print("=" * (5 + 1 + col_script + 1 + col_avg + 1 + col_runs))


if __name__ == "__main__":
    print_stats()
