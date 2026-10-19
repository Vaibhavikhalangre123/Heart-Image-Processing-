"""
Cardiology Coronary Artery Image Processing
============================================

Input modes:
1. Image: DICOM, TIFF, PNG, JPEG
2. Video: MP4, AVI, MOV, MKV
3. DICOM Cine Video: multi-frame grayscale DICOM

DICOM (single frame and cine) uses the SAME technique as video:
  rescale + photometric -> one shared intensity window -> every frame through
  process_video_frame() (CLAHE, black-hat, Otsu/adaptive threshold, soft mask,
  selective darkening). All processing lives in pipeline.py.

Academic/research visualization only.
Not for clinical diagnosis.
"""

import io
import sys
import time
import tempfile
import subprocess
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import (
    process_frame,
    process_video_frame,
    load_dicom_frames,
    robust_normalize16,
    suppress_background,
    to_uint16,
)


# =========================================================
# STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Coronary Image Processing",
    page_icon="🫀",
    layout="wide",
)

st.title("Cardiology Coronary Artery Image Processing")
st.caption("Academic/research demo — not for clinical diagnosis.")


# =========================================================
# SIDEBAR: DICOM SETTINGS (same video technique)
# =========================================================

with st.sidebar:
    st.header("DICOM settings")
    st.caption("DICOM images and cine use the video technique.")

    d_strength = st.slider(
        "Vessel darkening strength", 0.0, 1.0, 0.30, 0.01,
        help="Same 'strength' parameter as process_video_frame().",
    )
    d_bgsub = st.slider(
        "Background subtraction", 0.0, 1.0, 0.0, 0.05,
        help="0 = off (same as normal video).",
    )


# =========================================================
# DICOM LOADING (shared by Image and Cine modes)
# =========================================================

@st.cache_data(show_spinner=False, max_entries=2)
def load_dicom_cached(file_bytes):
    """Read a grayscale DICOM with rescale + photometric applied and ONE
    shared intensity window over all frames.

    Returns (frames_uint8 (N, H, W), fps).
    """
    frames8, fps = load_dicom_frames(file_bytes)
    return frames8, float(fps)


# =========================================================
# IMAGE HELPERS
# =========================================================

def to_uint16_gray(image):
    """Convert a standard image to 16-bit grayscale."""

    if image is None:
        raise ValueError("Could not decode the image.")

    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.ndim != 2:
        raise ValueError("Only 2D grayscale images are supported.")

    if image.dtype == np.uint8:
        return image.astype(np.uint16) * 257

    if image.dtype == np.uint16:
        return image.copy()

    image = np.asarray(image, dtype=np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=65535.0, neginf=0.0)

    low = float(image.min())
    high = float(image.max())

    if high <= low:
        return np.zeros(image.shape, dtype=np.uint16)

    # Preserve already valid nonnegative 16-bit intensities.
    if low >= 0 and high <= 65535:
        return image.astype(np.uint16)

    scaled = (image - low) * (65535.0 / (high - low))
    return np.clip(scaled, 0, 65535).astype(np.uint16)


def disp(image):
    """Convert image to 8-bit for display only."""

    image = np.asarray(image)

    if image.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    image = np.nan_to_num(
        image.astype(np.float32), nan=0.0, posinf=65535.0, neginf=0.0
    )

    low, high = np.percentile(image, (1.0, 99.5))

    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)

    scaled = (image - low) * (255.0 / (high - low))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def read_standard_image(uploaded_file):
    """Read a non-DICOM image upload as uint16 grayscale."""

    image = cv2.imdecode(
        np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:
        raise ValueError("Could not decode this image file.")

    return to_uint16_gray(image)


# =========================================================
# VIDEO HELPERS
# =========================================================

def make_video_writer(path, fps, width, height):
    """Create an MP4 intermediate writer."""

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
        isColor=True,
    )

    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            "Could not create video writer. Check OpenCV codec support."
        )

    return writer


def as_bgr_uint8(frame, width=None, height=None):
    """Prepare an output frame for OpenCV's color video writer."""

    if frame is None:
        raise ValueError("Processing returned an empty frame.")

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected a grayscale or BGR output frame.")

    if width is not None and height is not None:
        if frame.shape[:2] != (height, width):
            raise ValueError("Processed frame dimensions changed.")

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(frame)


def convert_to_h264(ffmpeg_exe, input_path, output_path):
    """Convert an intermediate video to browser-compatible H.264."""

    result = subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-i", str(input_path),
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        details = (
            result.stderr[-1800:]
            if result.stderr
            else "No FFmpeg details available."
        )
        raise RuntimeError(f"H.264 conversion failed: {details}")

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("H.264 conversion produced an empty file.")


# =========================================================
# GLOBAL INPUT OPTIONS
# =========================================================

mode = st.radio(
    "Choose input type",
    ["Image", "Video", "DICOM Cine Video"],
    horizontal=True,
)

invert = st.checkbox(
    "Invert vessel polarity",
    value=False,
    help="Use when vessels are BRIGHT in the source. Output vessels are black.",
)


# =========================================================
# MODE 1: IMAGE PROCESSING
# =========================================================

if mode == "Image":

    uploaded = st.file_uploader(
        "Upload DICOM, TIFF, PNG, or JPEG",
        type=["dcm", "dicom", "tif", "tiff", "png", "jpg", "jpeg"],
        key="image_upload",
    )

    if uploaded is None:
        st.info("Upload a single-frame image to begin.")

    else:
        try:
            is_dicom = uploaded.name.lower().endswith((".dcm", ".dicom"))

            # ---------------------------------------------
            # DICOM: new dark-vessel pipeline
            # ---------------------------------------------
            if is_dicom:
                frames8, _ = load_dicom_cached(uploaded.getvalue())

                if frames8.shape[0] != 1:
                    raise ValueError(
                        "This DICOM contains multiple frames. Select "
                        "DICOM Cine Video mode."
                    )

                raw8 = frames8[0]

                # Preview only: broad background suppression of the original.
                background = suppress_background(
                    robust_normalize16(to_uint16(raw8)),
                    kernel_size=21,
                    keep_original=0.80,
                )

                start = time.perf_counter_ns()
                video_result = process_video_frame(
                    raw8,
                    invert=invert,
                    strength=d_strength,
                    background_subtract=d_bgsub,
                )
                latency_ms = (time.perf_counter_ns() - start) / 1e6

                enhanced = np.ascontiguousarray(
                    video_result[..., 0] if video_result.ndim == 3
                    else video_result
                )

                st.success("DICOM loaded successfully.")
                st.subheader("Original vs Processed DICOM Image")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(raw8, caption="Original (rescaled)",
                             clamp=True, width="stretch")
                with col2:
                    st.image(disp(background), caption="Background Suppressed",
                             clamp=True, width="stretch")
                with col3:
                    st.image(enhanced, caption="Enhanced (video technique)",
                             clamp=True, width="stretch")

                ok, png = cv2.imencode(".png", enhanced)
                if ok:
                    st.download_button(
                        "Download enhanced PNG",
                        data=png.tobytes(),
                        file_name="enhanced_dicom.png",
                        mime="image/png",
                    )

            # ---------------------------------------------
            # Standard image: original pipeline.py
            # ---------------------------------------------
            else:
                raw16 = read_standard_image(uploaded)

                start = time.perf_counter_ns()
                background, enhanced = process_frame(raw16, invert=invert)
                latency_ms = (time.perf_counter_ns() - start) / 1e6

                if background is None or enhanced is None:
                    raise ValueError("The pipeline returned an empty result.")

                st.success("Standard image loaded successfully.")
                st.subheader("Original vs Processed Image")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(disp(raw16), caption="Original / Raw Image",
                             clamp=True, width="stretch")
                with col2:
                    st.image(disp(background), caption="Background Suppressed",
                             clamp=True, width="stretch")
                with col3:
                    st.image(disp(enhanced),
                             caption="Enhanced Coronary Artery",
                             clamp=True, width="stretch")

            # ---------------------------------------------
            # Performance
            # ---------------------------------------------
            st.subheader("Image Processing Performance")

            m1, m2 = st.columns(2)
            m1.metric("Core Processing Latency", f"{latency_ms:.2f} ms")
            m2.metric(
                "Equivalent FPS",
                f"{1000.0 / latency_ms:.1f}" if latency_ms > 0 else "N/A",
            )

            if latency_ms <= 36:
                st.success(
                    "This single run is within the 36 ms target. "
                    "Verify using repeated benchmarks."
                )
            else:
                st.warning(
                    "This run exceeds 36 ms. Use benchmark.py for "
                    "repeated performance measurements."
                )

        except Exception as exc:
            st.error(f"Image loading or processing failed: {exc}")


# =========================================================
# MODE 2: STANDARD VIDEO PROCESSING  (unchanged)
# =========================================================

elif mode == "Video":

    uploaded_video = st.file_uploader(
        "Upload MP4, AVI, MOV, or MKV video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload",
    )

    if uploaded_video is None:
        st.info("Upload a video to begin frame-by-frame processing.")

    else:
        try:
            import imageio_ffmpeg

            input_bytes = uploaded_video.getvalue()
            suffix = Path(uploaded_video.name).suffix.lower() or ".mp4"

            with tempfile.TemporaryDirectory() as temp_dir:
                temp = Path(temp_dir)

                input_path = temp / f"input{suffix}"
                intermediate = temp / "enhanced_intermediate.mp4"
                output_path = temp / "enhanced_video.mp4"

                input_path.write_bytes(input_bytes)

                cap = cv2.VideoCapture(str(input_path))

                if not cap.isOpened():
                    cap.release()
                    raise ValueError(
                        "Could not open this video. Try another MP4 file."
                    )

                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if not np.isfinite(fps) or fps <= 0:
                    fps = 25.0

                if width <= 0 or height <= 0:
                    cap.release()
                    raise ValueError("Could not read video dimensions.")

                writer = None
                processed = 0
                total_ms = 0.0

                progress = st.progress(0.0)
                status = st.empty()

                try:
                    writer = make_video_writer(
                        intermediate, fps, width, height
                    )

                    while True:
                        ok, frame = cap.read()

                        if not ok:
                            break

                        start = time.perf_counter_ns()
                        result = process_video_frame(frame, invert=invert)
                        total_ms += (time.perf_counter_ns() - start) / 1e6

                        result = as_bgr_uint8(
                            result, width=width, height=height
                        )

                        writer.write(result)
                        processed += 1

                        if expected_frames > 0:
                            progress.progress(
                                min(processed / expected_frames, 1.0)
                            )

                        if processed == 1 or processed % 10 == 0:
                            status.write(
                                f"Processed {processed} frame(s)..."
                            )

                finally:
                    cap.release()
                    if writer is not None:
                        writer.release()

                if processed == 0:
                    raise ValueError("No video frames were processed.")

                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                convert_to_h264(ffmpeg_exe, intermediate, output_path)

                output_bytes = output_path.read_bytes()

                st.success(f"Processed {processed} video frames.")

                st.subheader("Original vs Enhanced Video")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Original Video**")
                    st.video(input_bytes, width=400)

                with col2:
                    st.markdown("**Enhanced Video**")
                    st.video(output_bytes, format="video/mp4", width=400)

                avg_ms = total_ms / processed

                st.subheader("Video Processing Performance")

                m1, m2, m3 = st.columns(3)
                m1.metric("Processed Frames", processed)
                m2.metric("Average Processing / Frame", f"{avg_ms:.2f} ms")
                m3.metric(
                    "Processing FPS",
                    f"{1000.0 / avg_ms:.1f}" if avg_ms > 0 else "N/A",
                )

                if avg_ms <= 36:
                    st.success(
                        "Average enhancement time is within the 36 ms target. "
                        "Confirm using repeated benchmarks."
                    )
                else:
                    st.warning("Average enhancement time exceeds 36 ms.")

                progress.progress(1.0)
                status.success("Video processing complete.")

        except Exception as exc:
            st.error(f"Video processing failed: {exc}")


# =========================================================
# MODE 3: MULTI-FRAME GRAYSCALE DICOM CINE VIDEO
# =========================================================

elif mode == "DICOM Cine Video":

    uploaded_dicom = st.file_uploader(
        "Upload a multi-frame grayscale DICOM cine file",
        type=["dcm", "dicom"],
        key="dicom_cine_upload",
        help=(
            "Supports multi-frame grayscale DICOM. "
            "Color DICOM and single-frame DICOM are not supported in this mode."
        ),
    )

    if uploaded_dicom is None:
        st.info(
            "Upload a multi-frame DICOM cine file. "
            "Use Image mode for a single-frame DICOM."
        )

    else:
        try:
            import imageio_ffmpeg

            frames8, fps = load_dicom_cached(uploaded_dicom.getvalue())

            frame_count, height, width = frames8.shape

            if frame_count < 2:
                raise ValueError(
                    "Expected multi-frame grayscale DICOM. "
                    "Use Image mode for a single-frame DICOM."
                )

            if not np.isfinite(fps) or fps <= 0:
                fps = 15.0

            with tempfile.TemporaryDirectory() as temp_dir:
                temp = Path(temp_dir)

                original_intermediate = temp / "original_intermediate.mp4"
                enhanced_intermediate = temp / "enhanced_intermediate.mp4"
                original_h264_path = temp / "original_cine.mp4"
                enhanced_h264_path = temp / "enhanced_cine.mp4"

                original_writer = None
                enhanced_writer = None

                processed = 0
                total_ms = 0.0

                progress = st.progress(0.0)
                status = st.empty()

                try:
                    original_writer = make_video_writer(
                        original_intermediate, fps, width, height
                    )
                    enhanced_writer = make_video_writer(
                        enhanced_intermediate, fps, width, height
                    )

                    for index in range(frame_count):
                        gray = frames8[index]
                        original_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                        start = time.perf_counter_ns()
                        result = process_video_frame(
                            gray,
                            invert=invert,
                            strength=d_strength,
                            background_subtract=d_bgsub,
                        )
                        total_ms += (time.perf_counter_ns() - start) / 1e6

                        enhanced_bgr = as_bgr_uint8(
                            result, width=width, height=height
                        )

                        original_writer.write(original_bgr)
                        enhanced_writer.write(enhanced_bgr)

                        processed += 1
                        progress.progress(processed / frame_count)

                        if processed == 1 or processed % 10 == 0:
                            status.write(
                                f"Processed DICOM frame {processed} "
                                f"of {frame_count}..."
                            )

                finally:
                    if original_writer is not None:
                        original_writer.release()
                    if enhanced_writer is not None:
                        enhanced_writer.release()

                if processed == 0:
                    raise ValueError("No DICOM cine frames were processed.")

                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

                convert_to_h264(
                    ffmpeg_exe, original_intermediate, original_h264_path
                )
                convert_to_h264(
                    ffmpeg_exe, enhanced_intermediate, enhanced_h264_path
                )

                original_bytes = original_h264_path.read_bytes()
                enhanced_bytes = enhanced_h264_path.read_bytes()

                st.success(
                    f"Successfully processed {processed} DICOM cine frames."
                )

                st.subheader("Original vs Enhanced DICOM Cine")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Original DICOM Cine**")
                    st.video(
                        original_bytes, format="video/mp4", width="stretch"
                    )

                with col2:
                    st.markdown("**Enhanced DICOM Cine**")
                    st.video(
                        enhanced_bytes, format="video/mp4", width="stretch"
                    )

                st.download_button(
                    "Download enhanced cine (MP4)",
                    data=enhanced_bytes,
                    file_name="enhanced_cine.mp4",
                    mime="video/mp4",
                )

                avg_ms = total_ms / processed

                st.subheader("DICOM Cine Processing Performance")

                m1, m2, m3 = st.columns(3)
                m1.metric("Processed Frames", processed)
                m2.metric("Average Processing / Frame", f"{avg_ms:.2f} ms")
                m3.metric(
                    "Processing FPS",
                    f"{1000.0 / avg_ms:.1f}" if avg_ms > 0 else "N/A",
                )

                if avg_ms <= 36:
                    st.success(
                        "Average enhancement time is within the 36 ms target. "
                        "Confirm using repeated benchmarks."
                    )
                else:
                    st.warning(
                        "Average enhancement time exceeds the 36 ms target."
                    )

                progress.progress(1.0)
                status.success("DICOM cine processing complete.")

        except Exception as exc:
            st.error(f"DICOM cine processing failed: {exc}")
