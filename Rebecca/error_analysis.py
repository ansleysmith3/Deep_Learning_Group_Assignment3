"""
Person 3 — find the wrong predictions, group them, and save an example grid for the
"what went wrong" slide.

Groups (first match wins, checked in this order):
    length_wrong   predicted length != true length — model doesn't even know how many digits
    long_number    true length >= 4 — least-represented case in training data
    cropped_badly  crop content touches the image border — digit likely cut off by the box
    blurry         low-contrast / out-of-focus crop (cheap Laplacian-variance score)
    other          none of the above

These are cheap heuristics for a slide, not a validated taxonomy — eyeball the saved grid
and move an example by hand if a heuristic clearly got it wrong.

    python error_analysis.py --model artifacts/baseline_best.keras --split val
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ^ lets these scripts import data_pipeline/ and model_baseline_ximin_v2.py from the
# repo root no matter where this script is run from (it lives in Rebecca/).

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data_pipeline.config import MAX_DIGITS
from data_pipeline.load_data import decode, load_arrays, make_dataset

GROUP_ORDER = ["length_wrong", "long_number", "cropped_badly", "blurry", "other"]


def laplacian_variance(gray):
    """Low variance of the discrete Laplacian ~= blurry/flat. Pure numpy, no cv2/scipy."""
    lap = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1]
           + gray[1:-1, :-2] + gray[1:-1, 2:])
    return float(lap.var())


def border_ink_fraction(gray, margin=0.08):
    """Share of 'ink' (far from the crop's own median brightness) inside the outer rim.
    High values suggest a digit stroke is cut off by the crop box edge."""
    h, w = gray.shape
    m = max(1, int(round(margin * min(h, w))))
    mask = np.zeros_like(gray, dtype=bool)
    mask[:m, :] = mask[-m:, :] = mask[:, :m] = mask[:, -m:] = True
    baseline = np.median(gray)
    ink = np.abs(gray.astype(np.float32) - baseline) > 40
    return float(ink[mask].mean())


def classify(true_len, pred_len, gray, blur_threshold):
    if pred_len != true_len:
        return "length_wrong"
    if true_len >= 4:
        return "long_number"
    if border_ink_fraction(gray) > 0.35:
        return "cropped_badly"
    if laplacian_variance(gray) < blur_threshold:
        return "blurry"
    return "other"


def main(model_path, split, out_dir, examples_per_group):
    os.makedirs(out_dir, exist_ok=True)

    model = tf.keras.models.load_model(model_path)
    X, y_len, y_digits, files = load_arrays(split)
    ds = make_dataset(split, batch_size=64, shuffle=False)

    preds = model.predict(ds, verbose=0)
    pred_len = np.argmax(preds["length"], axis=1)
    pred_digits = np.column_stack(
        [np.argmax(preds[f"digit{i}"], axis=1) for i in range(1, MAX_DIGITS + 1)])

    gray_all = X.astype(np.float32).mean(axis=-1)  # (N, 64, 64)
    sample = gray_all[:: max(1, len(gray_all) // 500)]
    blur_threshold = float(np.median([laplacian_variance(g) for g in sample]))

    groups = {name: [] for name in GROUP_ORDER}
    correct = 0
    for i in range(len(X)):
        true_str = decode(y_len[i], y_digits[i])
        pred_str = decode(pred_len[i], pred_digits[i])
        if true_str == pred_str:
            correct += 1
            continue
        group = classify(y_len[i], pred_len[i], gray_all[i], blur_threshold)
        groups[group].append((i, true_str, pred_str))

    n_wrong = len(X) - correct
    print(f"{split}: {correct}/{len(X)} correct, {n_wrong} wrong")
    for name in GROUP_ORDER:
        print(f"  {name}: {len(groups[name])}")

    fig, axes = plt.subplots(len(GROUP_ORDER), examples_per_group,
                              figsize=(2 * examples_per_group, 2.2 * len(GROUP_ORDER)))
    for row, name in enumerate(GROUP_ORDER):
        examples = groups[name][:examples_per_group]
        for col in range(examples_per_group):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(examples):
                idx, true_str, pred_str = examples[col]
                ax.imshow(X[idx].astype(np.uint8))
                ax.set_title(f"true {true_str} / pred {pred_str}", fontsize=8)
            if col == 0:
                ax.text(-0.35, 0.5, f"{name}\n(n={len(groups[name])})", fontsize=9,
                        va="center", ha="right", transform=ax.transAxes)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"error_examples_{split}.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\nSaved {fig_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="artifacts/baseline_best.keras")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--out", default="experiments/error_analysis")
    p.add_argument("--examples_per_group", type=int, default=6)
    args = p.parse_args()
    main(args.model, args.split, args.out, args.examples_per_group)
