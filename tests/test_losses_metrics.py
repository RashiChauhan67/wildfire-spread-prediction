import numpy as np
import pytest
import tensorflow as tf

from wildfire_unet import (
    combined_focal_dice_loss,
    dice_loss,
    focal_loss,
    iou_metric,
    per_sample_iou,
)


def _mask(rows, cols, size=8):
    m = np.zeros((1, size, size, 1), dtype="float32")
    m[0, rows[0] : rows[1], cols[0] : cols[1], 0] = 1.0
    return m


def test_dice_loss_is_near_zero_for_perfect_prediction():
    y = _mask((2, 5), (2, 5))
    assert float(dice_loss(y, y)) < 0.05


def test_dice_loss_for_disjoint_prediction():
    # No overlap, 4 px each: dice = (0 + 1) / (8 + 1) with smooth=1, so loss = 1 - 1/9
    loss = float(dice_loss(_mask((0, 2), (0, 2)), _mask((5, 7), (5, 7))))
    assert loss == pytest.approx(1 - 1 / 9, abs=1e-5)


def test_focal_loss_penalizes_wrong_prediction_more():
    y = _mask((2, 5), (2, 5))
    wrong = 1.0 - y
    assert float(focal_loss(y, wrong)) > float(focal_loss(y, y * 0.99 + 0.005))


def test_combined_loss_is_sum_of_parts():
    y = _mask((2, 5), (2, 5))
    p = np.clip(y * 0.7 + 0.1, 0, 1)
    expected = float(focal_loss(y, p)) + float(dice_loss(y, p))
    assert float(combined_focal_dice_loss(y, p)) == pytest.approx(expected)


def test_batch_iou_metric_known_value():
    y_true = _mask((0, 4), (0, 4))  # 16 px
    y_pred = _mask((0, 4), (0, 2))  # 8 px, fully inside the label
    assert float(iou_metric(y_true, tf.constant(y_pred))) == pytest.approx(8 / 16)


def test_per_sample_iou_known_value():
    real = np.zeros((4, 4))
    real[:2, :2] = 1  # 4 px
    pred = np.zeros((4, 4))
    pred[:2, :1] = 1  # 2 px inside
    assert per_sample_iou(real, pred) == pytest.approx(0.5)


def test_per_sample_iou_empty_vs_empty_is_one():
    assert per_sample_iou(np.zeros((4, 4)), np.zeros((4, 4))) == 1.0


def test_per_sample_iou_missed_fire_is_zero():
    pred = np.zeros((4, 4))
    real = np.zeros((4, 4))
    real[0, 0] = 1
    assert per_sample_iou(real, pred) == 0.0
