import numpy as np

from wildfire_unet import build_attention_unet


def test_output_shape_and_range():
    model = build_attention_unet()
    out = model.predict(np.random.rand(2, 64, 64, 12).astype("float32"), verbose=0)

    assert out.shape == (2, 64, 64, 1)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_parameter_count_matches_original_notebook():
    # model.summary() in the original Kaggle notebook reported 32,455,045 total parameters.
    assert build_attention_unet().count_params() == 32_455_045
