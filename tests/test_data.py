import numpy as np
import pytest
import tensorflow as tf

from wildfire_unet import INPUT_FEATURES, OUTPUT_FEATURES, get_dataset
from wildfire_unet.data import normalize_feature


def _write_tfrecord(path, num_samples=3, seed=0):
    rng = np.random.default_rng(seed)
    with tf.io.TFRecordWriter(str(path)) as writer:
        for _ in range(num_samples):
            feats = {}
            for name in INPUT_FEATURES + OUTPUT_FEATURES:
                arr = rng.normal(5.0, 2.0, size=64 * 64).astype("float32")
                if name in ("PrevFireMask", "FireMask"):
                    arr = rng.choice([-1.0, 0.0, 1.0], size=64 * 64).astype("float32")
                feats[name] = tf.train.Feature(float_list=tf.train.FloatList(value=arr))
            example = tf.train.Example(features=tf.train.Features(feature=feats))
            writer.write(example.SerializeToString())


def test_normalize_feature_zscores_regular_channels():
    x = tf.constant(np.random.default_rng(1).normal(10, 3, (64, 64, 1)).astype("float32"))
    out = normalize_feature(x, "elevation").numpy()
    assert out.mean() == pytest.approx(0.0, abs=1e-4)
    assert out.std() == pytest.approx(1.0, abs=1e-3)


def test_normalize_feature_leaves_prev_fire_mask_unchanged():
    x = tf.constant(
        np.random.default_rng(1).choice([-1.0, 0.0, 1.0], (64, 64, 1)).astype("float32")
    )
    assert np.array_equal(normalize_feature(x, "PrevFireMask").numpy(), x.numpy())


def test_dataset_shapes_and_binary_labels(tmp_path):
    _write_tfrecord(tmp_path / "next_day_wildfire_spread_train_000.tfrecord", num_samples=5)

    ds = get_dataset(str(tmp_path / "next_day_wildfire_spread_train*"), batch_size=2, shuffle=False)
    inputs, labels = next(iter(ds))

    assert inputs.shape == (2, 64, 64, 12)
    assert labels.shape == (2, 64, 64, 1)
    # -1 (unlabeled) is mapped to 0; every label value is 0 or 1
    assert set(np.unique(labels.numpy())) <= {0.0, 1.0}
