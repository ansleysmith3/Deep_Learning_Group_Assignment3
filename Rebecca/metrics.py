"""
Person 3 — headline metric + per-digit accuracy, in one place so every experiment and the
final test run report numbers the same way.

    from metrics import evaluate_model
    summary = evaluate_model(model, val_ds)
    summary["whole_number_accuracy"]   # the headline metric for the deck
    summary["per_head_accuracy"]       # length / digit1..digit5, to explain WHY it's low

Whole-number accuracy only counts an image as correct if every digit AND the length are
right (Ximin's README calls this out: digit4/digit5 per-head accuracy looks ~100% mostly
because most numbers are short and those positions are blank, so per-head accuracy alone
is misleading — report it beside whole-number accuracy, never instead of it).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ^ lets these scripts import data_pipeline/ and model_baseline_ximin_v2.py from the
# repo root no matter where this script is run from (it lives in Rebecca/).

import numpy as np

from data_pipeline.load_data import HEAD_NAMES
from model_baseline_ximin_v2 import predict_house_numbers


def whole_number_accuracy(results):
    """results: list of (true_str, pred_str) from predict_house_numbers()."""
    correct = sum(1 for true_str, pred_str in results if true_str == pred_str)
    return correct / len(results), correct, len(results)


def per_digit_accuracy(model, dataset, num_batches=-1):
    """Accuracy of each of the 6 softmax heads on its own, independent of whole-number
    correctness, so a low whole-number score can be traced to the head causing it."""
    correct = {name: 0 for name in HEAD_NAMES}
    total = 0
    ds = dataset if num_batches is None else dataset.take(num_batches)
    for x_batch, y_true_batch in ds:
        preds = model.predict(x_batch, verbose=0)
        total += x_batch.shape[0]
        for name in HEAD_NAMES:
            pred_cls = np.argmax(preds[name], axis=1)
            true_cls = y_true_batch[name].numpy()
            correct[name] += int((pred_cls == true_cls).sum())
    return {name: correct[name] / total for name in HEAD_NAMES}


def evaluate_model(model, dataset, num_batches=-1):
    """One call for everything a deck slide or an experiment-log row needs."""
    results = predict_house_numbers(model, dataset, num_batches=num_batches)
    whole_acc, correct, n = whole_number_accuracy(results)
    per_head = per_digit_accuracy(model, dataset, num_batches=num_batches)
    return {
        "whole_number_accuracy": whole_acc,
        "correct": correct,
        "n": n,
        "per_head_accuracy": per_head,
        "results": results,
    }


if __name__ == "__main__":
    import tensorflow as tf
    from data_pipeline.load_data import make_dataset

    model = tf.keras.models.load_model("artifacts/baseline_best.keras")
    val_ds = make_dataset("val", batch_size=64, shuffle=False)
    summary = evaluate_model(model, val_ds)
    print(f"Whole-number accuracy: {summary['whole_number_accuracy']:.4f} "
          f"({summary['correct']}/{summary['n']})")
    print("Per-head accuracy:")
    for name, acc in summary["per_head_accuracy"].items():
        print(f"  {name}: {acc:.4f}")
