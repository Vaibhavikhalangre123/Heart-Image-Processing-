import argparse
from pathlib import Path
import cv2
import numpy as np


def curve_points(cx, cy, rx, ry, start, end, wobble, n=250):
    t = np.linspace(start, end, n)
    x = cx + rx * np.cos(t) + wobble * np.sin(3*t)
    y = cy + ry * np.sin(t) + wobble * np.cos(2*t)
    return np.stack([x, y], axis=1).astype(np.int32)


def generate(index, h=512, w=512, seed=1000):
    rng = np.random.default_rng(seed + index)
    yy, xx = np.mgrid[0:h, 0:w]

    # Lung-field background
    cx, cy = w/2, h/2
    lung = np.exp(-(((xx-w*0.32)/(w*0.30))**2 + ((yy-h*0.50)/(h*0.43))**2))
    lung += np.exp(-(((xx-w*0.68)/(w*0.30))**2 + ((yy-h*0.50)/(h*0.43))**2))
    img = 9000 + 8500 * lung

    # Spine-like central broad structure
    spine = np.exp(-((xx-w*0.50)/(w*0.055))**2) * (0.35 + 0.65*np.exp(-((yy-h*0.52)/(h*0.45))**2))
    img += 13000 * spine

    # Rib-like arcs
    for k in range(8):
        y0 = h*(0.15 + k*0.095)
        x0 = w*0.50
        rr = w*(0.33 + 0.015*k)
        theta = np.linspace(np.deg2rad(205), np.deg2rad(335), 220)
        x = x0 + rr*np.cos(theta)
        y = y0 + rr*0.42*np.sin(theta)
        pts = np.stack([x, y], axis=1).astype(np.int32)
        cv2.polylines(img, [pts], False, 12000, thickness=5, lineType=cv2.LINE_AA)

    # Synthetic coronary tree
    vessel = np.zeros((h, w), np.uint8)
    main = curve_points(w*0.47, h*0.49, w*0.22, h*0.20, np.deg2rad(205), np.deg2rad(345), 7)
    cv2.polylines(vessel, [main], False, 255, 4, cv2.LINE_AA)

    branches = [
        (w*0.50, h*0.50, w*0.17, h*0.18, 3.5),
        (w*0.51, h*0.51, w*0.14, h*0.12, 3.0),
        (w*0.54, h*0.50, w*0.18, h*0.10, 3.0),
        (w*0.44, h*0.52, w*0.12, h*0.20, 2.5),
        (w*0.46, h*0.50, w*0.20, h*0.07, 2.4),
    ]
    for i, (x0,y0,rx,ry,th) in enumerate(branches):
        pts = curve_points(x0,y0,rx,ry,np.deg2rad(185+i*5),np.deg2rad(285+i*6),2.5)
        cv2.polylines(vessel, [pts], False, 255, int(th), cv2.LINE_AA)

    vessel_blur = cv2.GaussianBlur(vessel, (5,5), 0).astype(np.float32)
    img += vessel_blur * 115

    # Acquisition variation and noise
    gain = rng.uniform(0.75, 1.35)
    img *= gain
    img += rng.normal(0, rng.uniform(800, 1800), (h,w))
    img = np.clip(img, 0, 65535)

    # Add a few moving/metal-like distractors
    for _ in range(3):
        x = int(rng.integers(20, w-20))
        y = int(rng.integers(20, h-20))
        cv2.circle(img, (x,y), int(rng.integers(2,6)), int(rng.integers(20000,50000)), -1)

    return img.astype(np.uint16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--out", default="data/synthetic")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(1, args.count+1):
        img = generate(i)
        cv2.imwrite(str(out / f"frame_{i:03d}.tif"), img)
    print(f"Generated {args.count} synthetic 16-bit TIFF frames in {out}")


if __name__ == "__main__":
    main()
