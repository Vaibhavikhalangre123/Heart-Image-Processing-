import argparse
import platform
import time
from pathlib import Path
import cv2
import numpy as np

from src.pipeline import read_gray16, process_frame


def benchmark(raw, warmup, iterations):
    for _ in range(warmup):
        process_frame(raw)

    samples = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        t0 = time.perf_counter_ns()
        process_frame(raw)
        samples[i] = (time.perf_counter_ns() - t0) / 1e6

    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input")
    p.add_argument("--input-dir")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iterations", type=int, default=500)
    args = p.parse_args()

    if args.input:
        raw = read_gray16(args.input)
    elif args.input_dir:
        paths = sorted(Path(args.input_dir).glob("*.tif"))
        if not paths:
            raise SystemExit("No TIFF images found.")
        raw = read_gray16(str(paths[0]))
    else:
        raise SystemExit("Provide --input or --input-dir")

    samples = benchmark(raw, args.warmup, args.iterations)
    avg = float(samples.mean())
    mn = float(samples.min())
    mx = float(samples.max())
    fps = 1000.0 / avg

    print("\n=== CORONARY IMAGE PIPELINE BENCHMARK ===")
    print(f"Resolution          : {raw.shape[1]} x {raw.shape[0]}")
    print(f"Input dtype         : {raw.dtype}")
    print(f"Warmup iterations   : {args.warmup}")
    print(f"Timed iterations    : {args.iterations}")
    print(f"Average latency     : {avg:.3f} ms")
    print(f"Minimum latency     : {mn:.3f} ms")
    print(f"Maximum latency     : {mx:.3f} ms")
    print(f"FPS (from average)  : {fps:.2f}")
    print(f"Python              : {platform.python_version()}")
    print(f"OS                  : {platform.platform()}")
    print(f"CPU                 : {platform.processor()}")
    print(f"OpenCV              : {cv2.__version__}")
    print(f"36 ms average PASS  : {'YES' if avg <= 36 else 'NO'}")
    print(f"36 ms maximum PASS  : {'YES' if mx <= 36 else 'NO'}")

    if mx <= 36:
        print("\nRESULT: Maximum measured latency meets the 36 ms requirement.")
    else:
        print("\nRESULT: Maximum measured latency does NOT meet the 36 ms requirement on this machine.")
        print("Do not claim compliance in the report unless the target evaluation hardware passes.")

if __name__ == "__main__":
    main()
