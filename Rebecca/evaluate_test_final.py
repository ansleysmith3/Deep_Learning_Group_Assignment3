"""
Person 3 — run the FINAL model on the held-out test set exactly ONCE, at the end (per the
assignment's "no peeking at test until the very end" rule). Point --model at whichever
checkpoint won the experiment log.

    python evaluate_test_final.py --model artifacts/baseline_best.keras
    python evaluate_test_final.py --model experiments/augmentation/model_best.keras
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ^ lets these scripts import data_pipeline/ and model_baseline_ximin_v2.py from the
# repo root no matter where this script is run from (it lives in Rebecca/).

import argparse
import json

import tensorflow as tf

from data_pipeline.load_data import make_dataset
from metrics import evaluate_model


def main(model_path, out_path):
    model = tf.keras.models.load_model(model_path)
    test_ds = make_dataset("test", batch_size=64, shuffle=False)
    summary = evaluate_model(model, test_ds, num_batches=-1)

    print(f"FINAL test whole-number accuracy: {summary['whole_number_accuracy']:.4f} "
          f"({summary['correct']}/{summary['n']})")
    print("Per-head test accuracy:")
    for name, acc in summary["per_head_accuracy"].items():
        print(f"  {name}: {acc:.4f}")

    with open(out_path, "w") as f:
        json.dump({
            "model_path": model_path,
            "whole_number_accuracy": summary["whole_number_accuracy"],
            "correct": summary["correct"],
            "n": summary["n"],
            "per_head_accuracy": summary["per_head_accuracy"],
        }, f, indent=2)
    print(f"\nSaved {out_path} — this is the number that goes on the final slide.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="artifacts/baseline_best.keras")
    p.add_argument("--out", default="experiments/final_test_results.json")
    args = p.parse_args()
    main(args.model, args.out)
