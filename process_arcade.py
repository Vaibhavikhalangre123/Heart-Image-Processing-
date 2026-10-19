#!/usr/bin/env python3
"""Run the coronary enhancement pipeline over an ARCADE image folder."""
import argparse
from pathlib import Path
import cv2
from src.pipeline import read_gray16, process_frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", default="outputs/arcade")
    p.add_argument("--invert", action="store_true")
    args = p.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    paths = sorted([*inp.glob("*.png"), *inp.glob("*.jpg"), *inp.glob("*.jpeg"), *inp.glob("*.tif"), *inp.glob("*.tiff")])
    if not paths:
        raise SystemExit(f"No images found in {inp}")
    out.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(paths)} images...")
    for i, path in enumerate(paths, 1):
        raw = read_gray16(str(path))
        _, enhanced = process_frame(raw, invert=args.invert)
        cv2.imwrite(str(out / f"{path.stem}_enhanced.png"), enhanced)
        if i % 50 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)}")
    print(f"Saved enhanced images to {out.resolve()}")


if __name__ == "__main__":
    main()
