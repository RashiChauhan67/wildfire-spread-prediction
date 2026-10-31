"""TFRecord input pipeline for the Next Day Wildfire Spread dataset."""

from __future__ import annotations

import tensorflow as tf

# 12 input channels, in the order they are stacked.
# 'elevation' is the digital elevation model (DEM) layer.
INPUT_FEATURES = [
    "elevation",
    "th",  # wind direction
    "vs",  # wind velocity
    "tmmn",  # minimum temperature
    "tmmx",  # maximum temperature
    "sph",  # specific humidity
    "pr",  # precipitation
    "pdsi",  # drought index
    "NDVI",  # vegetation
    "population",
    "erc",  # energy release component
    "PrevFireMask",  # fire mask of the previous day
]
OUTPUT_FEATURES = ["FireMask"]

PATCH_SIZE = 64


def normalize_feature(feature: tf.Tensor, feature_name: str) -> tf.Tensor:
    """Standardize one feature channel of a single sample.

    ``PrevFireMask`` is returned unchanged. Every other channel is z-scored
    using the mean and standard deviation of *that sample's own* 64x64 patch
    (not dataset-wide statistics), as in the original notebook.
    """
    if feature_name == "PrevFireMask":
        return feature
    return (feature - tf.reduce_mean(feature)) / (tf.math.reduce_std(feature) + 1e-6)


def parse_function(example_proto: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Parse one serialized ``tf.train.Example``.

    Returns:
        inputs: float tensor of shape ``[64, 64, 12]``.
        label: binary float tensor of shape ``[64, 64, 1]``. In the raw data,
            ``FireMask`` uses -1 for unlabeled pixels; these are treated as
            "no fire" (0), and every positive value becomes 1.
    """
    description = {
        name: tf.io.FixedLenFeature([PATCH_SIZE, PATCH_SIZE], tf.float32)
        for name in INPUT_FEATURES + OUTPUT_FEATURES
    }
    parsed = tf.io.parse_single_example(example_proto, description)

    channels = [
        normalize_feature(tf.expand_dims(parsed[name], axis=-1), name) for name in INPUT_FEATURES
    ]
    inputs = tf.concat(channels, axis=-1)

    label = tf.expand_dims(parsed["FireMask"], axis=-1)
    label = tf.where(label < 0, 0.0, label)
    label = tf.where(label > 0, 1.0, label)
    return inputs, label


def get_dataset(file_pattern: str, batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
    """Build a batched ``tf.data.Dataset`` from TFRecord files matching ``file_pattern``.

    Args:
        file_pattern: Glob for the TFRecord files, e.g. ``data/next_day_wildfire_spread_train*``.
        batch_size: Samples per batch.
        shuffle: Shuffle with a 1000-sample buffer (training). Pass ``False`` for
            evaluation so the sample order is deterministic.
    """
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)
    dataset = dataset.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
