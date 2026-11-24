"""
Step 1 of the data pipeline: look at the raw SVHN data before changing anything.

Answers the questions the rest of the pipeline depends on:
  - How many images and how many digits per image?  -> gives k (longest number)
  - How big are the images?                         -> shows why we must resize
  - What do the bounding boxes look like?           -> shows why we crop

Run:  python data_pipeline/explore_data.py
Set the SVHN_DIR environment variable if your data is not in data/SVHN.
"""
import json
import os
from collections import Counter

import numpy as np
from PIL import Image

from config import SVHN_DIR


def load_struct(split):
    """Read digitStruct.json for 'train' or 'test' -> list of {filename, boxes}."""
    with open(os.path.join(SVHN_DIR, split, "digitStruct.json")) as f:
        return json.load(f)


for split in ["train", "test"]:
    records = load_struct(split)
    lengths = [len(r["boxes"]) for r in records]

    # Image sizes: PIL only reads the header here, so this is fast.
    sizes = np.array([Image.open(os.path.join(SVHN_DIR, split, r["filename"])).size
                      for r in records])  # (width, height)

    # Boxes that stick out of the image (negative top/left or past the edge).
    out_of_bounds = 0
    for r, (w, h) in zip(records, sizes):
        for b in r["boxes"]:
            if (b["top"] < 0 or b["left"] < 0 or
                    b["left"] + b["width"] > w or b["top"] + b["height"] > h):
                out_of_bounds += 1
                break

    labels = Counter(int(b["label"]) for r in records for b in r["boxes"])

    print(f"=== {split} ===")
    print(f"images: {len(records)}")
    print(f"digits per image: {dict(sorted(Counter(lengths).items()))}")
    print(f"longest number (k): {max(lengths)}  -> "
          f"{[r['filename'] for r in records if len(r['boxes']) == max(lengths)]}")
    print(f"width  min/median/max: {sizes[:,0].min()} / {int(np.median(sizes[:,0]))} / {sizes[:,0].max()}")
    print(f"height min/median/max: {sizes[:,1].min()} / {int(np.median(sizes[:,1]))} / {sizes[:,1].max()}")
    print(f"images with a box outside the image: {out_of_bounds}")
    print(f"raw label counts (10 means digit 0): {dict(sorted(labels.items()))}")
    print()
