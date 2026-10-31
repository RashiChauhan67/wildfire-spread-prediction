"""Train the Attention U-Net and evaluate it on the test split.

Example:
    python scripts/train.py --data-dir data --epochs 10 --output-dir runs/baseline
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import tensorflow as tf

from wildfire_unet import (
    build_attention_unet,
    combined_focal_dice_loss,
    evaluate_model,
    get_dataset,
    iou_metric,
)

logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True, help="Folder with the TFRecord files.")
    p.add_argument("--output-dir", type=Path, default=Path("runs/default"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pattern = str(args.data_dir / "next_day_wildfire_spread_{}*")
    train_ds = get_dataset(pattern.format("train"), args.batch_size)
    val_ds = get_dataset(pattern.format("eval"), args.batch_size, shuffle=False)
    test_ds = get_dataset(pattern.format("test"), args.batch_size, shuffle=False)

    model = build_attention_unet()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss=combined_focal_dice_loss,
        metrics=["accuracy", iou_metric],
    )

    history = model.fit(train_ds, epochs=args.epochs, validation_data=val_ds)
    model.save(args.output_dir / "model.keras")

    metrics = evaluate_model(model, test_ds)
    logger.info("Test metrics: %s", metrics)

    (args.output_dir / "history.json").write_text(json.dumps(history.history, indent=2))
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, **metrics}, indent=2)
    )


if __name__ == "__main__":
    main()
