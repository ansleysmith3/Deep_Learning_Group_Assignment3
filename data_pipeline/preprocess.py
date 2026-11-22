"""
Step 2 of the data pipeline: turn raw SVHN images into fixed-size arrays + labels.

For every image:
  1. Find one box around ALL its digits (the "union" box).
  2. Grow that box by 30% so we don't cut off digit edges, and clip it to the image.
  3. Crop the image to that box and resize it to 64 x 64 x 3.
  4. Build labels: the number's length (1-5) and one label per digit position
     (0-9, or 10 = "no digit here").

Output: data/processed/{train,val,test}.npz with
  X        uint8   (N, 64, 64, 3)
  y_len    int32   (N,)      number of digits, 1..5
  y_digits int32   (N, 5)    digit per position, 10 = blank
  files    str     (N,)      original filename, for checking mistakes later

Run:  python data_pipeline/preprocess.py   (takes a few minutes)
"""
import json
import os

import numpy as np
from PIL import Image

from config import BLANK, IMG_SIZE, MARGIN, MAX_DIGITS, PROCESSED_DIR, SEED, SVHN_DIR, VAL_FRACTION


def crop_box(boxes, img_w, img_h):
    """Union of all digit boxes, grown by MARGIN, clipped to the image. Returns (left, top, right, bottom)."""
    left = min(b["left"] for b in boxes)
    top = min(b["top"] for b in boxes)
    right = max(b["left"] + b["width"] for b in boxes)
    bottom = max(b["top"] + b["height"] for b in boxes)

    # Grow by MARGIN in total on each axis: half on each side.
    pad_x = (right - left) * MARGIN / 2
    pad_y = (bottom - top) * MARGIN / 2

    # Clip: some boxes in digitStruct.json start at negative pixels or run past the edge.
    return (max(0, int(left - pad_x)), max(0, int(top - pad_y)),
            min(img_w, int(right + pad_x)), min(img_h, int(bottom + pad_y)))


def make_labels(boxes):
    """
    Length label + one label per position, e.g. house number 169:
        y_len = 3, y_digits = [1, 6, 9, 10, 10]
    SVHN stores the digit 0 as label 10, so we map 10 -> 0 first.
    Then 10 is free to mean "blank" for positions past the end of the number.
    """
    digits = [0 if int(b["label"]) == 10 else int(b["label"]) for b in boxes]
    padded = digits + [BLANK] * (MAX_DIGITS - len(digits))
    return len(digits), padded


def process_split(split):
    folder = os.path.join(SVHN_DIR, split)
    with open(os.path.join(folder, "digitStruct.json")) as f:
        records = json.load(f)

    X, y_len, y_digits, files = [], [], [], []
    skipped = []
    for r in records:
        boxes = r["boxes"]
        if len(boxes) > MAX_DIGITS:  # only train/29930.png (6 digits)
            skipped.append(r["filename"])
            continue

        img = Image.open(os.path.join(folder, r["filename"])).convert("RGB")
        img = img.crop(crop_box(boxes, *img.size)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

        length, digits = make_labels(boxes)
        X.append(np.asarray(img, dtype=np.uint8))
        y_len.append(length)
        y_digits.append(digits)
        files.append(r["filename"])

    print(f"{split}: kept {len(X)}, skipped {skipped}")
    return (np.stack(X), np.array(y_len, dtype=np.int32),
            np.array(y_digits, dtype=np.int32), np.array(files))


def save(name, X, y_len, y_digits, files):
    path = os.path.join(PROCESSED_DIR, f"{name}.npz")
    np.savez(path, X=X, y_len=y_len, y_digits=y_digits, files=files)
    print(f"saved {name}: X {X.shape}, y_len {y_len.shape}, y_digits {y_digits.shape} -> {path}")


if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Train -> split off a fixed validation set (same seed = same split for the whole team).
    X, y_len, y_digits, files = process_split("train")
    idx = np.random.default_rng(SEED).permutation(len(X))
    n_val = int(len(X) * VAL_FRACTION)
    val, tr = idx[:n_val], idx[n_val:]
    save("train", X[tr], y_len[tr], y_digits[tr], files[tr])
    save("val", X[val], y_len[val], y_digits[val], files[val])

    # Test stays untouched until the final model.
    save("test", *process_split("test"))
