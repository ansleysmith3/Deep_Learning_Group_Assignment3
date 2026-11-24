"""
Person 3 — run ONE experiment that changes a single knob from Ximin's baseline, train it,
score it with the headline whole-number accuracy on val, and log the result.

Examples (pick ONE change per run — that's the point of an ablation):
    python run_experiment.py --name augmentation --change "data augmentation" --augment
    python run_experiment.py --name dropout05 --change "dropout 0.3 -> 0.5" --dropout 0.5
    python run_experiment.py --name extra_conv --change "4th conv block (256 filters)" --extra_conv_block
    python run_experiment.py --name lr_sched --change "LR schedule + early stopping" --lr_schedule --early_stopping
    python run_experiment.py --name l2 --change "L2 regularization 1e-4" --l2 1e-4
    python run_experiment.py --name smoke --fast_debug   # ~20 batches, to sanity-check the code

Each run writes experiments/<name>/model_best.keras + history.json, and appends one row to
experiments/experiment_log.csv (created with a header + the baseline's own numbers the first
time this runs, if the log doesn't exist yet) — that CSV is the "what worked / what didn't"
slide and the exam-prep answer to "how did you pick the final model".

Note: input size (64x64) and MAX_DIGITS are baked into data_pipeline/preprocess.py's output,
so changing them needs re-running preprocessing, not a flag here.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ^ lets these scripts import data_pipeline/ and model_baseline_ximin_v2.py from the
# repo root no matter where this script is run from (it lives in Rebecca/).

import argparse
import csv
import json
import os
import time

import tensorflow as tf
from tensorflow.keras import layers, models

from data_pipeline.config import MAX_DIGITS
from data_pipeline.load_data import HEAD_NAMES, make_dataset
from metrics import evaluate_model

LOG_PATH = "experiments/experiment_log.csv"
LOG_FIELDS = [
    "run_name", "change", "settings", "val_whole_number_accuracy",
    "val_correct", "val_n", "epochs_run", "train_seconds", "notes",
]

DEFAULTS = {
    "run_name": "unnamed",
    "change": None,
    "notes": "",
    "batch_size": 64,
    "learning_rate": 1e-3,
    "epochs": 15,
    "conv_filters": [32, 64, 128],
    "dense_units": 256,
    "dropout_rate": 0.3,
    "augment": False,
    "extra_conv_block": False,
    "l2": 0.0,
    "lr_schedule": False,
    "early_stopping": False,
    "fast_debug": False,
}


def build_model(config):
    inputs = layers.Input(shape=(64, 64, 3), name="image_input")
    x = inputs

    if config["augment"]:
        # Never a horizontal flip here: it mirrors digit shapes into something meaningless
        # (and would reverse left-right digit order for multi-digit numbers).
        x = layers.RandomRotation(0.05)(x)
        x = layers.RandomTranslation(0.05, 0.05)(x)
        x = layers.RandomContrast(0.1)(x)

    conv_filters = list(config["conv_filters"])
    if config["extra_conv_block"]:
        conv_filters = conv_filters + [256]

    reg = tf.keras.regularizers.l2(config["l2"]) if config["l2"] else None
    for idx, filters in enumerate(conv_filters, start=1):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu",
                           kernel_regularizer=reg, name=f"conv{idx}_1")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu",
                           kernel_regularizer=reg, name=f"conv{idx}_2")(x)
        x = layers.MaxPooling2D(2, name=f"pool{idx}")(x)
        x = layers.BatchNormalization(name=f"bn{idx}")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(config["dense_units"], activation="relu", name="shared_dense")(x)
    x = layers.Dropout(config["dropout_rate"], name="shared_dropout")(x)

    outputs = {"length": layers.Dense(MAX_DIGITS + 1, activation="softmax", name="length")(x)}
    for i in range(1, MAX_DIGITS + 1):
        outputs[f"digit{i}"] = layers.Dense(11, activation="softmax", name=f"digit{i}")(x)

    return models.Model(inputs, outputs, name=f"SVHN_{config['run_name']}")


def seed_log_with_baseline():
    """First call creates the log with Ximin's own baseline run as row 1, so every
    experiment is judged against the number the team already has."""
    append_log({
        "run_name": "baseline",
        "change": "baseline (no changes)",
        "settings": json.dumps({
            "conv_filters": [32, 64, 128], "dropout_rate": 0.3, "augment": False,
            "extra_conv_block": False, "l2": 0.0, "lr_schedule": False, "early_stopping": False,
        }),
        "val_whole_number_accuracy": "0.7620",
        "val_correct": 2545,
        "val_n": 3340,
        "epochs_run": 15,
        "train_seconds": "2546.3",
        "notes": "Ximin's baseline (model_baseline_ximin_v2.py). Best checkpoint epoch 5 "
                 "(val_loss~1.35); overfits after that. From README_model_baseline-v2.md.",
    })


def append_log(row):
    os.makedirs("experiments", exist_ok=True)
    is_new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def run(config):
    if not os.path.exists(LOG_PATH):
        seed_log_with_baseline()

    out_dir = os.path.join("experiments", config["run_name"])
    os.makedirs(out_dir, exist_ok=True)

    train_ds = make_dataset("train", batch_size=config["batch_size"], shuffle=True)
    val_ds = make_dataset("val", batch_size=config["batch_size"], shuffle=False)

    eval_batches = -1
    if config["fast_debug"]:
        train_ds, val_ds = train_ds.take(20), val_ds.take(5)
        eval_batches = 5

    model = build_model(config)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss={name: "sparse_categorical_crossentropy" for name in HEAD_NAMES},
        metrics={name: "accuracy" for name in HEAD_NAMES},
    )

    callbacks = [tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(out_dir, "model_best.keras"),
        monitor="val_loss", save_best_only=True, verbose=0)]
    if config["lr_schedule"]:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1))
    if config["early_stopping"]:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True))

    epochs = 3 if config["fast_debug"] else config["epochs"]
    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                         callbacks=callbacks, verbose=1)
    elapsed = time.time() - start
    epochs_run = len(history.history["loss"])

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history.history, f, indent=2)

    best_model = tf.keras.models.load_model(os.path.join(out_dir, "model_best.keras"))
    summary = evaluate_model(best_model, val_ds, num_batches=eval_batches)

    append_log({
        "run_name": config["run_name"],
        "change": config["change"] or config["run_name"],
        "settings": json.dumps({k: v for k, v in config.items()
                                 if k not in ("run_name", "change", "notes")}),
        "val_whole_number_accuracy": f"{summary['whole_number_accuracy']:.4f}",
        "val_correct": summary["correct"],
        "val_n": summary["n"],
        "epochs_run": epochs_run,
        "train_seconds": f"{elapsed:.1f}",
        "notes": config["notes"],
    })

    print(f"\n[{config['run_name']}] whole-number val accuracy: "
          f"{summary['whole_number_accuracy']:.4f} ({summary['correct']}/{summary['n']}) "
          f"vs. baseline 0.7620")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", dest="run_name", required=True)
    p.add_argument("--change", default=None, help="short label for the log")
    p.add_argument("--notes", default="")
    p.add_argument("--dropout", type=float, dest="dropout_rate")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--extra_conv_block", action="store_true")
    p.add_argument("--l2", type=float)
    p.add_argument("--lr_schedule", action="store_true")
    p.add_argument("--early_stopping", action="store_true")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--fast_debug", action="store_true")
    args = p.parse_args()

    config = dict(DEFAULTS)
    for k, v in vars(args).items():
        if v is not None:
            config[k] = v
    return config


if __name__ == "__main__":
    run(parse_args())
