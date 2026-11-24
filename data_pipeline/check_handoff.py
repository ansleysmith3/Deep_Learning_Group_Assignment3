"""
Sanity checks on the processed data + proof the hand-off plugs into a 6-head Keras model.
The tiny model here is NOT our model (that's Person 2's); it only checks names and shapes line up.

Run:  python data_pipeline/check_handoff.py
"""
import numpy as np
import tensorflow as tf

from config import BLANK, MAX_DIGITS
from load_data import HEAD_NAMES, decode, load_arrays, make_dataset

# 1. Labels are consistent: positions past the length are blank, positions inside are real digits.
for split in ["train", "val", "test"]:
    X, y_len, y_digits, files = load_arrays(split)
    for n, digits in zip(y_len, y_digits):
        assert all(d != BLANK for d in digits[:n]) and all(d == BLANK for d in digits[n:])
    print(f"{split}: {X.shape}, lengths {np.bincount(y_len)[1:]}, labels OK")

# 2. The example from the assignment PDF: 33402.png is the house number 169.
for split in ["train", "val"]:
    X, y_len, y_digits, files = load_arrays(split)
    hit = np.where(files == "33402.png")[0]
    if len(hit):
        i = hit[0]
        print(f"33402.png ({split}): y_len={y_len[i]}, y_digits={y_digits[i]}, decoded='{decode(y_len[i], y_digits[i])}'")

# 3. One batch from the tf.data pipeline.
x, y = next(iter(make_dataset("train", batch_size=8, shuffle=True)))
print("batch x:", x.shape, x.dtype, f"range [{x.numpy().min():.2f}, {x.numpy().max():.2f}]")
print("batch y:", {k: v.shape for k, v in y.items()})

# 4. A throwaway 6-head model trains on it for a few steps.
inp = tf.keras.Input((64, 64, 3))
h = tf.keras.layers.Conv2D(16, 3, activation="relu")(inp)
h = tf.keras.layers.GlobalAveragePooling2D()(h)
outputs = {"length": tf.keras.layers.Dense(MAX_DIGITS + 1, activation="softmax", name="length")(h)}
for name in HEAD_NAMES[1:]:
    outputs[name] = tf.keras.layers.Dense(11, activation="softmax", name=name)(h)
model = tf.keras.Model(inp, outputs)
model.compile("adam", loss={k: "sparse_categorical_crossentropy" for k in HEAD_NAMES})
model.fit(make_dataset("train", batch_size=64).take(5), epochs=1, verbose=2)
print("hand-off OK: 6 named heads train on make_dataset()")
