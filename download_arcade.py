#!/usr/bin/env python3
"""Download and prepare the MedOtter/ARCADE coronary dataset.

Downloads both ARCADE subsets (stenosis and syntax) from Hugging Face and
exports the images/masks as ordinary PNG files for this project.
"""
import argparse
from pathlib import Path
from datasets import load_dataset

REPO_ID = "MedOtter/ARCADE"
SUBSETS = ("stenosis", "syntax")
SPLITS = ("train", "val", "test")


def save_split(ds, subset: str, split: str, output_root: Path, limit: int | None = None):
    out = output_root / subset / split
    image_dir = out / "images"
    mask_dir = out / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    n = len(ds) if limit is None else min(limit, len(ds))
    for i in range(n):
        item = ds[i]
        filename = Path(str(item.get("file_name", f"{i+1}.png"))).stem
        item["image"].convert("L").save(image_dir / f"{filename}.png")
        item["mask"].convert("L").save(mask_dir / f"{filename}.png")
        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f"  {subset}/{split}: {i+1}/{n}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/ARCADE_processed")
    p.add_argument("--subset", choices=["all", *SUBSETS], default="all")
    p.add_argument("--split", choices=["all", *SPLITS], default="all")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional number of images per selected split (useful for a quick test).")
    args = p.parse_args()

    subsets = SUBSETS if args.subset == "all" else (args.subset,)
    splits = SPLITS if args.split == "all" else (args.split,)
    output_root = Path(args.output)

    print(f"Hugging Face dataset: {REPO_ID}")
    print(f"Output: {output_root.resolve()}")

    for subset in subsets:
        print(f"\nLoading subset: {subset}")
        ds = load_dataset(REPO_ID, subset)
        for split in splits:
            if split not in ds:
                print(f"  Skipping missing split: {subset}/{split}")
                continue
            save_split(ds[split], subset, split, output_root, args.limit)

    print("\nARCADE download/conversion complete.")
    print(f"Images and masks: {output_root.resolve()}")


if __name__ == "__main__":
    main()
