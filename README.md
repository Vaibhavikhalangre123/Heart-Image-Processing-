# Cardiology Coronary Artery Image Processing

A real-time starter implementation for processing 16-bit grayscale cardiology X-ray/cine frames.

## Important scope

This project is an **image-processing research/demo pipeline**, not a clinically validated medical device. It is designed for an academic assignment and should not be used for diagnosis or patient care.

The pipeline uses classical OpenCV operations rather than a trained deep-learning model, which keeps the runtime low and makes the processing steps easy to explain and benchmark.

## Features

- 8-bit and 16-bit grayscale input; 16-bit is preserved internally and in the enhanced output.
- Background/anatomical structure suppression using multi-scale morphological background estimation.
- Dark/bright structure suppression through morphological top-hat/black-hat style vessel extraction.
- Local contrast enhancement using CLAHE.
- Multi-scale vessel enhancement.
- Optional fast rib/spine/background suppression mask.
- Raw / processed / enhanced comparison image.
- Synthetic, legally usable test dataset generated locally.
- Latency benchmark with average/minimum/maximum latency and FPS.
- Optional Streamlit interface.
- No patient data is included.

## Pipeline

Input frame
-> grayscale conversion
-> 16-bit normalization
-> robust denoising
-> local contrast normalization
-> multi-scale background estimation
-> vessel candidate extraction
-> anatomical/background suppression
-> vessel response fusion
-> percentile normalization
-> final 16-bit enhanced image

For coronary angiography, contrast-filled vessels can appear brighter than surrounding structures. The exact polarity can vary by acquisition system, so the implementation exposes `invert` and scale parameters.

## Project structure

```text
cardiology_coronary_image_processing/
├── src/
│   ├── pipeline.py
│   ├── app.py
│   └── generate_dataset.py
├── data/
│   └── synthetic/
├── outputs/
├── tests/
├── benchmark.py
├── requirements.txt
└── README.md
```

## 1. Install Python

Recommended: Python 3.10 or 3.11 on Windows.

Check:

```powershell
python --version
```

## 2. Create virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again.

## 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Generate the synthetic dataset

```powershell
python src/generate_dataset.py --count 20
```

This creates legal synthetic 16-bit angiography-like frames under `data/synthetic/`.

## 5. Run the command-line processor

```powershell
python src/pipeline.py --input data/synthetic/frame_001.tif --output outputs/frame_001_enhanced.tif --save-comparison outputs/frame_001_comparison.png
```

The enhanced output is saved as a 16-bit TIFF.

## 6. Run the interactive application

```powershell
streamlit run src/app.py
```

Open the URL shown by Streamlit, normally:

```text
http://localhost:8501
```

Upload a 16-bit grayscale TIFF/PNG image. The application displays:

1. Original/raw image
2. Background-suppressed image
3. Enhanced coronary-vessel image

It also displays measured core-pipeline latency.

## 7. Benchmark the 36 ms requirement

Use:

```powershell
python benchmark.py --input data/synthetic/frame_001.tif --warmup 50 --iterations 500
```

The benchmark reports:

- Average latency
- Minimum latency
- Maximum latency
- FPS
- Hardware configuration
- Whether average latency is <= 36 ms

### Important

The assignment requires **maximum processing latency <= 36 ms**, not only average latency. Therefore, the benchmark prints both average and maximum. If maximum latency is above 36 ms on your machine, optimize/repeat the test on the target evaluation hardware and report the actual result honestly.

The benchmark measures only the processing pipeline, excluding disk I/O and UI rendering. This is the cleanest interpretation of "input frame received until final processed frame produced" for the algorithm itself.

## 8. Test all images

```powershell
python benchmark.py --input-dir data/synthetic --warmup 50 --iterations 100
```

## 9. Tests

```powershell
python -m pytest -q
```

## Dataset information

The included dataset is procedurally generated synthetic data. It contains 16-bit grayscale images with:
- vessel-like branching curves
- rib-like arcs
- spine-like central structures
- lung-field gradients
- Gaussian noise
- contrast variation

No patient-identifiable information is included.

For a stronger academic evaluation, add a legally licensed/public coronary angiography dataset and document its license. Do not commit restricted patient data to GitHub.

## Model information

No trained model is required in this baseline. The approach is classical computer vision and is intentionally lightweight for the 36 ms target.

## Latency optimization notes

- OpenCV operations run in optimized native code.
- The benchmark avoids disk I/O inside the timed section.
- Images are converted to a compact floating representation only where required.
- Gaussian/morphological kernels are kept small.
- Streamlit visualization is outside the core latency benchmark.

For stricter real-time evaluation, use a fixed input resolution representative of the target cine system and report CPU model, RAM, OS, OpenCV version, Python version, and image resolution.

## Academic report sections to include

1. Problem statement
2. Dataset and licensing
3. Input characteristics
4. Preprocessing
5. Background suppression
6. Coronary vessel enhancement
7. 16-bit handling
8. Implementation
9. Latency benchmark
10. Results
11. Limitations
12. Future work

## Clinical limitation

Ribs, spine, lung fields and coronary vessels overlap anatomically. A purely classical filter cannot guarantee perfect anatomical separation in every patient/frame. For a stronger research system, a supervised vessel-segmentation model could be added, but that introduces dataset, training, inference latency and validation requirements.

## ARCADE dataset integration

This project can download and prepare the public MedOtter/ARCADE X-ray coronary angiography dataset from Hugging Face.

Dataset: https://huggingface.co/datasets/MedOtter/ARCADE

ARCADE contains 3,000 grayscale 512x512 X-ray coronary angiography images across two subsets: `stenosis` and `syntax`. Each subset has `train` (1,000), `val` (200), and `test` (300), with an image and mask. The dataset card lists CC0 1.0 Universal. See the Hugging Face dataset card for source and licensing details.

### Install

```powershell
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Download and convert the complete ARCADE dataset

```powershell
python download_arcade.py
```

This creates:

```text
data/ARCADE_processed/
  stenosis/
    train/images/
    train/masks/
    val/images/
    val/masks/
    test/images/
    test/masks/
  syntax/
    train/images/
    train/masks/
    val/images/
    val/masks/
    test/images/
    test/masks/
```

For a quick test before downloading everything:

```powershell
python download_arcade.py --subset stenosis --split train --limit 10
```

### Run the optimized pipeline on ARCADE

```powershell
python process_arcade.py --input-dir data/ARCADE_processed/stenosis/train/images --output-dir outputs/arcade/stenosis_train
```

### Benchmark the optimized pipeline

```powershell
python benchmark.py --input data/ARCADE_processed/stenosis/train/images/676.png --warmup 50 --iterations 500
```

Use a filename that actually exists in your downloaded folder if `676.png` is not present.

The benchmark measures only the core processing function, not disk I/O or image display. Run it on the target hardware before reporting latency/FPS.
