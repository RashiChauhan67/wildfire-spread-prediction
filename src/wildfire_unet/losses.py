"""Loss functions for the heavily imbalanced fire / no-fire segmentation task."""

from __future__ import annotations

import tensorflow as tf


def focal_loss(y_true, y_pred, alpha: float = 0.25, gamma: float = 2.0):
    """Binary focal loss (mean over all pixels)."""
    y_true = tf.cast(y_true, tf.float32)
    eps = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    weight = alpha * tf.pow(1 - y_pred, gamma) * y_true + (1 - alpha) * tf.pow(y_pred, gamma) * (
        1 - y_true
    )
    return tf.reduce_mean(weight * cross_entropy)


def dice_loss(y_true, y_pred, smooth: float = 1.0):
    """Soft Dice loss over the whole batch.

    NOTE: the saved Kaggle notebook calls ``dice_loss`` but its definition is not
    present in the saved file. This is a standard soft-Dice implementation; the
    smoothing constant of the original run is unknown.
    """
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)


def combined_focal_dice_loss(y_true, y_pred):
    """Focal loss + Dice loss, the training objective used in the notebook."""
    return focal_loss(y_true, y_pred) + dice_loss(y_true, y_pred)
