import numpy as np
from src.pipeline import process_frame


def test_output_is_16bit():
    x = np.random.default_rng(1).integers(0, 65535, (256, 256), dtype=np.uint16)
    bg, out = process_frame(x)
    assert bg.dtype == np.uint16
    assert out.dtype == np.uint16
    assert out.shape == x.shape


def test_pipeline_does_not_return_nan():
    x = np.full((128, 128), 30000, dtype=np.uint16)
    _, out = process_frame(x)
    assert np.isfinite(out.astype(np.float32)).all()
