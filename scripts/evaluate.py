"""Evaluate a saved model on the test split.

Example:
    python scripts/evaluate.py --data-dir data --model runs/baseline/model.keras
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from wildfire_unet import combined_focal_dice_loss, evaluate_model, get_dataset, iou_metric


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    model = tf.keras.models.load_model(
        args.model,
        custom_objects={
            "combined_focal_dice_loss": combined_focal_dice_loss,
            "iou_metric": iou_metric,
        },
    )
    test_ds = get_dataset(
        str(args.data_dir / "next_day_wildfire_spread_test*"), args.batch_size, shuffle=False
    )
    print(json.dumps(evaluate_model(model, test_ds), indent=2))


if __name__ == "__main__":
    main()
