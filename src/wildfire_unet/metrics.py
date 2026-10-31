"""IoU metrics.

Two different IoU conventions appear in this project, and they give different numbers:

* ``iou_metric`` is the batch-level pixel IoU Keras reports during training and validation.
* ``per_sample_iou`` / ``evaluate_model`` compute IoU per 64x64 sample and average over samples.
  A sample with no fire in both the label and the prediction scores 1.0 (the convention of the
  original notebook), which raises the mean when many samples contain no fire.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def iou_metric(y_true, y_pred):
    """Batch-level pixel IoU at a 0.5 threshold (Keras metric)."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(tf.cast(tf.math.logical_or(y_true > 0.5, y_pred > 0.5), tf.float32))
    return tf.math.divide_no_nan(intersection, union)


def per_sample_iou(real_mask: np.ndarray, predicted_mask: np.ndarray) -> float:
    """IoU of one sample. Returns 1.0 when both masks are empty."""
    real = np.asarray(real_mask) > 0
    predicted = np.asarray(predicted_mask) > 0
    union = np.logical_or(real, predicted).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(real, predicted).sum() / union)


def evaluate_model(model, dataset: tf.data.Dataset, threshold: float = 0.5) -> dict[str, float]:
    """Evaluate a trained model on a batched ``(inputs, labels)`` dataset.

    Returns:
        ``mean_sample_iou``: mean of per-sample IoU (empty-vs-empty counts as 1.0).
        ``mean_sample_iou_fire_only``: the same, restricted to samples whose label contains fire.
        ``pixel_iou``: IoU computed from intersections and unions summed over the whole dataset.
    """
    sample_ious: list[float] = []
    fire_only: list[float] = []
    inter_total = 0
    union_total = 0

    for inputs, labels in dataset:
        predictions = (model.predict(inputs, verbose=0) > threshold).astype(np.uint8)[..., 0]
        labels = labels.numpy()[..., 0] > 0
        for real, pred in zip(labels, predictions, strict=True):
            iou = per_sample_iou(real, pred)
            sample_ious.append(iou)
            if real.any():
                fire_only.append(iou)
            inter_total += np.logical_and(real, pred).sum()
            union_total += np.logical_or(real, pred).sum()

    return {
        "mean_sample_iou": float(np.mean(sample_ious)),
        "mean_sample_iou_fire_only": float(np.mean(fire_only)) if fire_only else float("nan"),
        "pixel_iou": float(inter_total / union_total) if union_total else 1.0,
        "num_samples": len(sample_ious),
    }
