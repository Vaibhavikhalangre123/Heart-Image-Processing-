"""
Coronary angiography enhancement: standard images, videos and DICOM (.dcm).

DICOM uses the SAME technique as video: every DICOM frame is rescaled once,
converted to uint8 with ONE shared intensity window, and then passed through
process_video_frame() (CLAHE -> black-hat -> Otsu/adaptive threshold ->
soft mask -> selective darkening -> light sharpening).

Entry points
  process_frame(img16, ...)        -> (background_uint16, final_uint16)   images
  process_video_frame(frame, ...)  -> uint8 BGR frame                      videos
  load_dicom_frames(path_or_bytes) -> (frames_uint8 (N,H,W), fps)          DICOM
  process_dicom_frames(frames8)    -> enhanced uint8 stack (N,H,W)         DICOM
  process_dicom_file(path)         -> (raw, enhanced) or enhanced stack    DICOM
  enhance_black_vessels(frame, ..) -> uint8 gray frame (dark vessels)
  calibrate_black_state(frames)    -> locked thresholds for a whole video

Academic/research visualization only. Not for clinical diagnosis.
"""

import argparse
import io
import time
from pathlib import Path

import cv2
import numpy as np
import pydicom


# ============================================================
# REUSABLE VIDEO CLAHE
# Defined once at module level so it is not rebuilt per frame.
# ============================================================

VIDEO_CLAHE = cv2.createCLAHE(
    clipLimit=1.3,
    tileGridSize=(8, 8)
)


# ============================================================
# IMAGE READING
# ============================================================

def read_gray16(path: str) -> np.ndarray:
    """Read a standard image (PNG, JPEG, TIFF, ...) as uint16 grayscale."""

    img = cv2.imread(str(Path(path)), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.dtype == np.uint16:
        return img

    if img.dtype == np.uint8:
        return img.astype(np.uint16) * 257

    values = img.astype(np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        if info.min < 0:
            lo, hi = float(values.min()), float(values.max())
            if hi <= lo:
                return np.zeros(img.shape, dtype=np.uint16)
            values = (values - lo) / (hi - lo)
        else:
            values = values / max(float(info.max), 1.0)
    else:
        lo, hi = float(values.min()), float(values.max())
        if lo < 0 or hi > 1:
            if hi <= lo:
                return np.zeros(img.shape, dtype=np.uint16)
            values = (values - lo) / (hi - lo)

    return np.clip(values * 65535.0, 0, 65535).astype(np.uint16)


# ============================================================
# NORMALIZATION AND DISPLAY CONVERSION
# ============================================================

def robust_normalize16(
    img16: np.ndarray,
    low: float = 1.0,
    high: float = 99.5
) -> np.ndarray:
    """Percentile normalization for a grayscale image."""

    image = img16.astype(np.float32)
    lo, hi = np.percentile(image, (low, high))

    if hi <= lo + 1:
        return img16.copy()

    normalized = (image - lo) * (65535.0 / (hi - lo))
    return np.clip(normalized, 0, 65535).astype(np.uint16)


def to_uint8(
    image: np.ndarray,
    percentiles=(0.5, 99.5)
) -> np.ndarray:
    """Convert an image to uint8 for OpenCV processing/display."""

    if image.dtype == np.uint8:
        return image.copy()

    values = image.astype(np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    if image.dtype == np.uint16:
        return np.clip(values / 257.0, 0, 255).astype(np.uint8)

    lo, hi = np.percentile(values, percentiles)

    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)

    return np.clip(
        (values - lo) * (255.0 / (hi - lo)),
        0, 255
    ).astype(np.uint8)


def to_uint16(image: np.ndarray) -> np.ndarray:
    """Convert uint8 grayscale output to uint16."""

    return image.astype(np.uint16) * 257


# ============================================================
# CLAHE
# ============================================================

def clahe16(
    img16: np.ndarray,
    clip_limit: float = 1.2,
    tile_grid=(8, 8)
) -> np.ndarray:
    """Mild local contrast enhancement for a uint16 image."""

    x8 = to_uint8(img16)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid
    )

    return to_uint16(clahe.apply(x8))


# ============================================================
# BACKGROUND SUPPRESSION
# ============================================================

def suppress_background(
    img16: np.ndarray,
    kernel_size: int = 21,
    keep_original: float = 0.80
) -> np.ndarray:
    """Estimate broad background and gently blend a detail-enhanced
    image with the original."""

    x8 = to_uint8(img16)

    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    background = cv2.morphologyEx(x8, cv2.MORPH_OPEN, kernel)
    residual = cv2.subtract(x8, background)

    detail_weight = 1.0 - float(np.clip(keep_original, 0.0, 1.0))

    output = cv2.addWeighted(
        x8, 1.0 - detail_weight,
        residual, detail_weight,
        0
    )

    return to_uint16(output)


# ============================================================
# MULTI-SCALE VESSEL RESPONSE (legacy path)
# ============================================================

def vessel_response(
    img16: np.ndarray,
    invert: bool = False
) -> np.ndarray:
    """Morphological black-hat response.

    Default polarity detects dark, narrow structures.
    invert=True is intended for bright vessel structures.
    """

    x8 = to_uint8(img16)

    if invert:
        x8 = cv2.bitwise_not(x8)

    response = np.zeros_like(x8)

    for size in (5, 7, 11):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (size, size)
        )
        blackhat = cv2.morphologyEx(x8, cv2.MORPH_BLACKHAT, kernel)
        np.maximum(response, blackhat, out=response)

    response = cv2.GaussianBlur(response, (3, 3), 0.6)

    return to_uint16(response)


def clean_vessel_response(vessel16: np.ndarray) -> np.ndarray:
    """Suppress weak responses while retaining stronger candidates."""

    vessel8 = to_uint8(vessel16)
    positive = vessel8[vessel8 > 0]

    if positive.size == 0:
        return np.zeros_like(vessel16, dtype=np.uint16)

    threshold = float(np.percentile(positive, 65))
    upper = float(np.percentile(positive, 95))

    if upper <= threshold:
        upper = threshold + 1.0

    response = np.clip(
        (vessel8.astype(np.float32) - threshold)
        * (255.0 / (upper - threshold)),
        0, 255
    ).astype(np.uint8)

    response = cv2.GaussianBlur(response, (3, 3), 0.6)

    return to_uint16(response)


def suppress_large_structures(
    vessel16: np.ndarray,
    raw16: np.ndarray
) -> np.ndarray:
    """Weight vessel candidates by local image detail."""

    vessel8 = to_uint8(vessel16)
    raw8 = to_uint8(raw16)

    blurred = cv2.GaussianBlur(raw8, (0, 0), 2.0)
    detail = cv2.absdiff(raw8, blurred)
    detail = cv2.normalize(detail, None, 0, 255, cv2.NORM_MINMAX)

    # Avoid fully removing a vessel where contrast is weak.
    weight = 0.40 + 0.60 * (detail.astype(np.float32) / 255.0)
    result = vessel8.astype(np.float32) * weight

    return to_uint16(np.clip(result, 0, 255).astype(np.uint8))


def enhance_vessel_layer(
    base16: np.ndarray,
    vessel16: np.ndarray,
    strength: float = 18.0,
    invert: bool = False
) -> np.ndarray:
    """Gently enhance pixels with stronger vessel responses."""

    base8 = to_uint8(base16)
    vessel8 = to_uint8(vessel16)

    positive = vessel8[vessel8 > 0]

    if positive.size == 0:
        return to_uint16(base8)

    threshold = float(np.percentile(positive, 55))
    upper = float(np.percentile(positive, 95))

    if upper <= threshold:
        upper = threshold + 1.0

    mask = np.clip(
        (vessel8.astype(np.float32) - threshold) / (upper - threshold),
        0.0, 1.0
    )
    mask = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0.6)

    base = base8.astype(np.float32)
    amount = float(np.clip(strength, 0.0, 60.0))

    if invert:
        result = base + amount * mask * (255.0 - base) / 255.0
    else:
        result = base - amount * mask

    return to_uint16(np.clip(result, 0, 255).astype(np.uint8))


def sharpen_vessels(
    img16: np.ndarray,
    amount: float = 0.12
) -> np.ndarray:
    """Apply mild unsharp masking."""

    x8 = to_uint8(img16)
    blurred = cv2.GaussianBlur(x8, (0, 0), 1.0)

    sharp = cv2.addWeighted(
        x8, 1.0 + amount,
        blurred, -amount,
        0
    )

    return to_uint16(sharp)


# ============================================================
# BLACK VESSELS WITHOUT BRIGHTENING THE IMAGE
#
#   1. Estimate and suppress the background (closing keeps big dark
#      vessels out of the estimate; only slow shading is divided out,
#      so overall brightness is unchanged - no contrast stretch).
#   2. Multi-scale black-hat vessel response on the corrected image.
#   3. Threshold the response (Otsu-based floor + hysteresis: weak thin
#      vessels survive only if they touch a strong one) -> vessel mask.
#   4. Darken pixels only where the mask says vessel:
#          result = base * (1 - strength * mask)
#   5. No CLAHE after darkening (it would bring background texture back);
#      only a very light unsharp mask.
# ============================================================

def _find_roi(gray8):
    """Bounding box (x0, y0, x1, y1) of the real image area, found from the
    strong intensity step at the edge of the dark vignette/border."""

    h, w = gray8.shape
    g = gray8.astype(np.float32)

    def edges(profile, n):
        p = cv2.GaussianBlur(profile.reshape(1, -1), (0, 0), 3).ravel()
        d = np.gradient(p)
        a = int(n * 0.30)
        i0 = int(np.argmax(d[:a]))                  # dark -> bright
        i1 = int(n - a + np.argmin(d[n - a:]))      # bright -> dark
        strength = min(d[i0], -d[i1])
        return i0, i1, strength

    cols = g[h // 4:3 * h // 4].mean(axis=0)
    rows = g[:, w // 4:3 * w // 4].mean(axis=1)
    x0, x1, sx = edges(cols, w)
    y0, y1, sy = edges(rows, h)

    if sx < 0.15 or sy < 0.15 or x1 - x0 < 0.4 * w or y1 - y0 < 0.4 * h:
        return 0, 0, w, h                            # no border found

    pad = max(2, int(0.008 * min(h, w)))
    return x0 + pad, y0 + pad, x1 - pad, y1 - pad


def _gray_float(frame):
    """Frame -> float32 gray on a 0..255 scale, without contrast stretching."""

    if frame.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        frame = cv2.cvtColor(frame, code)

    if frame.dtype == np.uint16:
        return frame.astype(np.float32) / 257.0

    if frame.dtype == np.uint8:
        return frame.astype(np.float32)

    return to_uint8(frame).astype(np.float32)


def enhance_black_vessels(
    frame,
    invert=False,          # True = source vessels are BRIGHT; output vessels are still black
    state=None,            # one dict per video (see calibrate_black_state)
    strength=0.88,         # 0..1 how black the vessels become (1 = fully black)
    sensitivity=1.0,       # >1 finds more thin vessels (and more noise); <1 is stricter
    shading=0.6,           # 0..1 how much slow background shading is flattened
    sharpen=0.15,          # light unsharp mask; 0 = off
    min_depth=0.55,        # strong seeds must be this fraction as deep as calibrated vessels
):
    """Return an 8-bit grayscale frame: same brightness, vessels darkened."""

    if frame is None or frame.size == 0:
        raise ValueError("Input frame is empty.")

    if state is None:
        state = {}

    gray = _gray_float(frame)
    if invert:
        gray = 255.0 - gray

    if "roi" not in state:
        state["roi"] = _find_roi(np.clip(gray, 0, 255).astype(np.uint8))

    x0, y0, x1, y1 = [int(v) for v in state["roi"]]
    crop = gray[y0:y1, x0:x1].astype(np.float32)
    h, w = crop.shape
    rs = min(h, w) / 1024.0

    denoised = cv2.GaussianBlur(crop, (0, 0), 0.8)

    # Clip black speckle for DETECTION only (the output keeps real pixels)
    floor_val = float(np.percentile(crop, 1.0))
    det = cv2.GaussianBlur(np.maximum(crop, floor_val), (0, 0), 0.9)
    logdet = np.log1p(np.maximum(det, 0)).astype(np.float32)

    # ---- STEP 1: estimate and suppress the background -------------
    max_vessel_px = 28 * rs
    ds = 4
    small = cv2.resize(
        logdet, (max(16, w // ds), max(16, h // ds)), interpolation=cv2.INTER_AREA
    )
    k = max(5, int(2 * max_vessel_px / ds) | 1)
    small = cv2.copyMakeBorder(small, k, k, k, k, cv2.BORDER_REFLECT)
    bg = cv2.morphologyEx(
        small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    bg = cv2.GaussianBlur(bg, (0, 0), max(2.0, k / 3.0))[k:-k, k:-k]
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    flat = (logdet - bg).astype(np.float32)          # ~0 background, <0 vessels

    # Base image: divide out slow shading only. Dividing by a median-normalised
    # background keeps the average brightness unchanged.
    bg_lin = np.expm1(bg)
    shade = bg_lin / (float(np.median(bg_lin)) + 1e-6)
    base = denoised / np.power(np.maximum(shade, 0.05), shading)
    base = np.clip(base, 0, 255).astype(np.float32)

    # ---- STEP 2: black-hat vessel response ------------------------
    # 100*exp(flat) = background-corrected image (background ~100)
    corrected = (100.0 * np.exp(flat)).astype(np.float32)
    response = np.zeros_like(corrected)
    for size in (5, 9, 15, 25):
        size = max(3, int(round(size * rs)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        np.maximum(
            response,
            cv2.morphologyEx(corrected, cv2.MORPH_BLACKHAT, kernel),
            out=response
        )
    response = cv2.GaussianBlur(response, (0, 0), 0.8)   # ~ % contrast below background

    # ---- STEP 3: threshold the response -> vessel mask ------------
    ref_now = float(np.percentile(response, 99.5)) + 1e-6
    ref = state["ref"] if state.get("locked") else ref_now
    r8 = np.clip(response / ref * 255.0, 0, 255).astype(np.uint8)

    if state.get("locked"):
        floor, seed, hi = state["floor"], state["seed"], state["hi"]
    else:
        otsu, _ = cv2.threshold(r8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu = float(otsu)
        floor = max(2.0, 0.70 * otsu)
        seed = max(floor + 1.0, otsu)
        # pixels at/above this response level become fully dark (strength)
        hi = max(1.25 * seed, floor + 1.0)
        if state.get("calibrating"):
            neg_now = abs(float(np.percentile(flat, 0.5))) + 1e-6
            state.setdefault("cal", []).append(
                (ref_now, floor, seed, max(hi, floor + 1.0), neg_now)
            )

    floor = floor / max(sensitivity, 1e-3)
    neg = (
        state["neg"] if state.get("locked")
        else abs(float(np.percentile(flat, 0.5))) + 1e-6
    )
    depth = np.clip(-flat / neg, 0, 1)

    weak = (r8 > floor).astype(np.uint8)
    strong = ((r8 > seed) & (depth > min_depth)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
    keep = np.zeros(n, np.uint8)
    keep[np.unique(lab[strong > 0])] = 1
    keep[0] = 0
    keep[stats[:, cv2.CC_STAT_AREA] < max(20, 2.5e-5 * h * w)] = 0
    mask = keep[lab].astype(np.float32)

    soft = np.clip((r8.astype(np.float32) - floor) / max(hi - floor, 1.0), 0, 1)
    vessel = cv2.GaussianBlur(soft * mask, (0, 0), 0.8)

    # ---- STEP 4: darken ONLY where the mask indicates vessels -----
    strength = float(np.clip(strength, 0.0, 1.0))
    result = base * (1.0 - strength * np.clip(vessel * 1.15, 0, 1))
    result = np.clip(result, 0, 255)

    # ---- STEP 5: no CLAHE after darkening (light unsharp only) ----
    if sharpen > 0:
        blur = cv2.GaussianBlur(result, (0, 0), 1.0)
        result = np.clip(result * (1.0 + sharpen) - blur * sharpen, 0, 255)

    full = gray.copy()                   # outside the image area: untouched
    full[y0:y1, x0:x1] = result

    return np.clip(full, 0, 255).astype(np.uint8)


def calibrate_black_state(frames, sample_count=8, **kwargs):
    """Lock thresholds for a whole video from a few evenly spaced frames, so
    frames with little contrast agent are not over-enhanced and nothing
    flickers."""

    n = len(frames)
    idx = np.unique(np.linspace(0, n - 1, min(sample_count, n)).astype(int))
    tmp = {"calibrating": True}

    for i in idx:
        enhance_black_vessels(frames[int(i)], state=tmp, **kwargs)

    ref, floor, seed, hi, neg = max(tmp["cal"], key=lambda t: t[4])  # deepest-vessel frame

    return {
        "roi": tmp["roi"], "ref": ref, "floor": floor, "seed": seed,
        "hi": hi, "neg": neg, "locked": True,
    }


# ============================================================
# IMAGE PIPELINE
# ============================================================

def enhance_coronaries(
    img16: np.ndarray,
    invert: bool = False
) -> np.ndarray:
    """Return the enhanced uint16 coronary angiography image."""

    _, final = process_frame(img16, invert=invert)
    return final


def process_frame(
    img16: np.ndarray,
    invert: bool = False,
    black_vessels: bool = True,
    strength: float = 0.88,
    sensitivity: float = 1.0
):
    """Return background-corrected and final enhanced uint16 images.

    black_vessels=True (default): vessels are darkened only where detected,
    overall brightness is preserved and no CLAHE is applied afterwards.
    black_vessels=False: previous CLAHE-based behaviour.
    """

    if not black_vessels:
        return process_frame_legacy(img16, invert=invert)

    x = robust_normalize16(img16)
    x = cv2.GaussianBlur(x, (3, 3), 0.6)
    background = suppress_background(x, kernel_size=21, keep_original=0.80)

    final8 = enhance_black_vessels(
        img16, invert=invert, state={},
        strength=strength, sensitivity=sensitivity
    )
    return background, to_uint16(final8)


def process_frame_legacy(
    img16: np.ndarray,
    invert: bool = False
):
    """Previous pipeline (CLAHE before vessel detection). Kept for comparison."""

    x = robust_normalize16(img16)
    x = cv2.GaussianBlur(x, (3, 3), 0.6)
    x = clahe16(x, clip_limit=1.2, tile_grid=(8, 8))

    background = suppress_background(x, kernel_size=21, keep_original=0.80)

    vessels = vessel_response(x, invert=invert)
    vessels = clean_vessel_response(vessels)
    vessels = suppress_large_structures(vessels, x)

    fused = enhance_vessel_layer(
        background,
        vessels,
        strength=18.0,
        invert=invert
    )

    final = sharpen_vessels(fused, amount=0.12)

    return background, final


# ============================================================
# VIDEO FRAME PROCESSING WITH THRESHOLDING
# ============================================================

def process_video_frame(
    frame: np.ndarray,
    invert: bool = False,
    strength: float = 0.30,
    background_subtract: float = 0.0
) -> np.ndarray:
    """Process one video frame using CLAHE, Otsu + adaptive thresholding,
    a soft candidate mask and gentle selective enhancement.

    Returns an 8-bit BGR frame at the original resolution.

    background_subtract:
        Optional broad-background suppression (0.0 to 1.0).
        Default 0.0 = off.
    """

    if frame is None or frame.size == 0:
        raise ValueError("Input video frame is empty.")

    # 1. Convert to grayscale.
    if frame.ndim == 3:
        if frame.shape[2] == 4:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        elif frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported frame shape: {frame.shape}")
    elif frame.ndim == 2:
        gray = frame
    else:
        raise ValueError(f"Unsupported frame shape: {frame.shape}")

    # 2. Convert to 8-bit.
    gray8 = to_uint8(gray)

    # 3. Mild denoising.
    denoised = cv2.GaussianBlur(gray8, (3, 3), 0.6)

    # 4. Local contrast enhancement.
    enhanced = VIDEO_CLAHE.apply(denoised)

    # Optional broad-background suppression.
    background_subtract = float(np.clip(background_subtract, 0.0, 1.0))
    if background_subtract > 0.0:
        h, w = enhanced.shape
        sigma = max(8.0, min(h, w) / 18.0)
        broad_background = cv2.GaussianBlur(
            enhanced, (0, 0), sigmaX=sigma, sigmaY=sigma
        ).astype(np.float32)
        reference = float(np.median(broad_background))
        flattened = enhanced.astype(np.float32) - background_subtract * (
            broad_background - reference
        )
        enhanced = np.clip(flattened, 0, 255).astype(np.uint8)

    # 5. Build a vessel-response image.
    # Black-hat emphasizes dark line-like structures.
    # For bright vessels, invert the input before black-hat.
    detection_input = cv2.bitwise_not(enhanced) if invert else enhanced

    response = np.zeros_like(detection_input)

    for size in (5, 7, 11):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        blackhat = cv2.morphologyEx(detection_input, cv2.MORPH_BLACKHAT, kernel)
        np.maximum(response, blackhat, out=response)

    response = cv2.GaussianBlur(response, (3, 3), 0.6)

    # 6. Threshold the vessel response (Otsu OR adaptive).
    _, otsu_mask = cv2.threshold(
        response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    adaptive_mask = cv2.adaptiveThreshold(
        response, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, -2
    )
    candidate_mask = cv2.bitwise_or(otsu_mask, adaptive_mask)

    # 7. Reject zero-response pixels to reduce unrelated background.
    candidate_mask[response == 0] = 0

    # 8. Create a soft mask from selected vessel-response pixels.
    if not np.any(candidate_mask):
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    selected = response[candidate_mask > 0]
    low = float(np.percentile(selected, 20))
    high = float(np.percentile(selected, 95))

    if high <= low:
        high = low + 1.0

    soft_mask = np.clip(
        (response.astype(np.float32) - low) / (high - low),
        0.0, 1.0
    )
    soft_mask *= (candidate_mask > 0).astype(np.float32)
    soft_mask = cv2.GaussianBlur(soft_mask, (3, 3), 0.6)

    # 9. Selective enhancement.
    base = enhanced.astype(np.float32)
    strength = float(np.clip(strength, 0.0, 1.0))

    if invert:
        # Bright-vessel polarity.
        result = base + strength * soft_mask * (255.0 - base)
    else:
        # Dark-vessel polarity.
        result = base - strength * soft_mask * base

    result = np.clip(result, 0, 255).astype(np.uint8)

    # 10. Gentle sharpening.
    blurred = cv2.GaussianBlur(result, (0, 0), 0.7)
    result = cv2.addWeighted(result, 1.08, blurred, -0.08, 0)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


# ============================================================
# DICOM / DCM: SAME VIDEO TECHNIQUE, APPLIED FRAME BY FRAME
#
#   1. Read pixel array, apply RescaleSlope/Intercept and
#      PhotometricInterpretation (MONOCHROME1 is inverted).
#   2. Convert to uint8 with ONE window for the whole stack
#      (no frame-to-frame brightness changes).
#   3. Run every frame through process_video_frame().
# ============================================================

def _dicom_fps(ds, default=15.0):
    for tag in ("RecommendedDisplayFrameRate", "CineRate"):
        value = getattr(ds, tag, None)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            return value
    frame_time = getattr(ds, "FrameTime", None)
    try:
        frame_time = float(frame_time)
        if np.isfinite(frame_time) and frame_time > 0:
            return 1000.0 / frame_time
    except (TypeError, ValueError):
        pass
    return default


def load_dicom_frames(source):
    """Read a grayscale DICOM (path, bytes or file-like).

    Returns (frames_uint8, fps) where frames_uint8 has shape (N, H, W);
    a single-frame DICOM gives N = 1.
    """
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(bytes(source))
    elif not hasattr(source, "read"):
        source = str(source)

    try:
        ds = pydicom.dcmread(source, force=True)
    except Exception as exc:
        raise ValueError(f"Cannot read DICOM file: {exc}") from exc

    if not hasattr(ds, "PixelData"):
        raise ValueError("DICOM file has no pixel data.")
    if int(getattr(ds, "SamplesPerPixel", 1)) != 1:
        raise ValueError("Only grayscale DICOM is supported.")

    try:
        px = np.asarray(ds.pixel_array)
    except Exception as exc:
        raise ValueError(
            "Could not decode DICOM pixels. Compressed DICOM may need "
            f"pylibjpeg or python-gdcm installed: {exc}"
        ) from exc

    if px.ndim == 2:
        px = px[None]
    elif px.ndim != 3:
        raise ValueError(f"Unsupported DICOM pixel shape: {px.shape}")

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    if not np.isfinite(slope) or not np.isfinite(intercept):
        raise ValueError("DICOM rescale slope/intercept must be finite.")

    # One shared window for every frame (estimated on a subsample).
    sample = px[:, ::4, ::4].astype(np.float32) * slope + intercept
    sample = np.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(sample, (0.5, 99.5))
    if high <= low:
        raise ValueError("DICOM has no usable intensity range.")

    scale = 255.0 / (high - low)
    out = np.empty(px.shape, dtype=np.uint8)
    for first in range(0, px.shape[0], 8):          # chunked: low peak memory
        last = min(first + 8, px.shape[0])
        v = px[first:last].astype(np.float32) * slope + intercept
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        out[first:last] = np.clip((v - low) * scale, 0, 255).astype(np.uint8)

    if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper() == "MONOCHROME1":
        out = 255 - out

    return out, _dicom_fps(ds)


def process_dicom_frames(
    frames8: np.ndarray,
    invert: bool = False,
    strength: float = 0.30,
    background_subtract: float = 0.0
) -> np.ndarray:
    """Run every uint8 gray frame (N, H, W) through process_video_frame().
    Returns an enhanced uint8 gray stack with the same shape."""

    out = np.empty_like(frames8)
    for i, frame in enumerate(frames8):
        result = process_video_frame(
            frame,
            invert=invert,
            strength=strength,
            background_subtract=background_subtract
        )
        out[i] = result[..., 0] if result.ndim == 3 else result
    return out


def process_dicom_file(
    path,
    invert: bool = False,
    strength: float = 0.30,
    background_subtract: float = 0.0
):
    """Single-frame DICOM -> (original_uint8, enhanced_uint8)
    Multi-frame DICOM  -> enhanced uint8 stack (frames, rows, cols)"""

    frames8, _ = load_dicom_frames(path)
    enhanced = process_dicom_frames(
        frames8, invert=invert, strength=strength,
        background_subtract=background_subtract
    )
    if len(frames8) == 1:
        return frames8[0], enhanced[0]
    return enhanced


def process_dicom(path, invert=False, **kwargs):
    """Backward-compatible alias."""
    return process_dicom_file(path, invert=invert, **kwargs)


# ============================================================
# COMPARISON IMAGE
# ============================================================

def save_comparison(
    raw16: np.ndarray,
    bg16: np.ndarray,
    enhanced16: np.ndarray,
    path
):
    """Save original, background-corrected and enhanced images."""

    panels = [
        to_uint8(raw16),
        to_uint8(bg16),
        to_uint8(enhanced16)
    ]

    comparison = cv2.hconcat(panels)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), comparison):
        raise OSError(f"Could not save comparison image: {output_path}")


# ============================================================
# COMMAND-LINE IMAGE PROCESSING
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Coronary angiography image enhancement"
    )

    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--save-comparison", help="Optional comparison path")
    parser.add_argument(
        "--invert", action="store_true", help="Use bright-vessel polarity"
    )
    parser.add_argument(
        "--strength", type=float, default=0.88,
        help="How black vessels become, 0..1 (default 0.88)"
    )
    parser.add_argument(
        "--sensitivity", type=float, default=1.0,
        help=">1 finds more thin vessels (and noise), <1 is stricter"
    )
    parser.add_argument(
        "--legacy", action="store_true",
        help="Use the previous CLAHE-based pipeline instead"
    )

    args = parser.parse_args()

    raw = read_gray16(args.input)

    start = time.perf_counter()
    background, enhanced = process_frame(
        raw,
        invert=args.invert,
        black_vessels=not args.legacy,
        strength=args.strength,
        sensitivity=args.sensitivity
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), enhanced):
        raise OSError(f"Could not save enhanced image: {output_path}")

    if args.save_comparison:
        save_comparison(raw, background, enhanced, args.save_comparison)

    print(f"Input shape: {raw.shape}, dtype={raw.dtype}")
    print(f"Output shape: {enhanced.shape}, dtype={enhanced.dtype}")
    print(f"Processing time: {elapsed_ms:.2f} ms")
    print(f"Saved enhanced image: {output_path}")


if __name__ == "__main__":
    main()
