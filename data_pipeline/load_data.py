"""
Step 3 of the data pipeline: the hand-off to the model (Person 2) and experiments (Person 3).

    from data_pipeline.load_data import make_dataset
    train_ds = make_dataset("train", batch_size=64, shuffle=True)
    val_ds   = make_dataset("val",   batch_size=64)

Each batch is (x, y):
    x  float32 (batch, 64, 64, 3), pixels scaled to [0, 1]
    y  dict of 6 int32 label arrays, one per softmax head:
         "length"  (batch,)  1..5       -> softmax with MAX_DIGITS + 1 = 6 classes (0..5)
         "digit1"  (batch,)  0..9 or 10 -> softmax with 11 classes
         ...
         "digit5"  (batch,)
The model's output layers must use these exact names so Keras matches each label to its head.
Use sparse_categorical_crossentropy for every head (labels are integers, not one-hot).
"""
import os

import numpy as np
import tensorflow as tf

try:
    from .config import MAX_DIGITS, PROCESSED_DIR
except ImportError:  # run from inside data_pipeline/
    from config import MAX_DIGITS, PROCESSED_DIR

HEAD_NAMES = ["length"] + [f"digit{i + 1}" for i in range(MAX_DIGITS)]


def load_arrays(split):
    """Raw numpy arrays for 'train', 'val' or 'test': X (uint8), y_len, y_digits, files."""
    d = np.load(os.path.join(PROCESSED_DIR, f"{split}.npz"))
    return d["X"], d["y_len"], d["y_digits"], d["files"]


def to_label_dict(y_len, y_digits):
    """(N,) and (N, 5) arrays -> {"length": ..., "digit1": ..., ..., "digit5": ...}."""
    labels = {"length": y_len}
    for i in range(MAX_DIGITS):
        labels[f"digit{i + 1}"] = y_digits[:, i]
    return labels


def make_dataset(split, batch_size=64, shuffle=False):
    X, y_len, y_digits, _ = load_arrays(split)
    ds = tf.data.Dataset.from_tensor_slices((X, to_label_dict(y_len, y_digits)))
    if shuffle:
        ds = ds.shuffle(len(X), seed=42, reshuffle_each_iteration=True)
    # Scale pixels after batching: cheaper, and X stays uint8 (4x less memory) until then.
    ds = ds.batch(batch_size).map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    return ds.prefetch(tf.data.AUTOTUNE)


def decode(y_len, y_digits):
    """Turn labels or predictions back into the house number string, e.g. (3, [1,6,9,10,10]) -> '169'."""
    return "".join(str(d) for d in y_digits[:y_len])
