"""
SVHN data pipeline (Person 1), ALL IN ONE FILE: raw Street View photos -> 64x64 images + 6 labels.

This is an optional, self-contained copy of the small scripts in this folder, pasted together
in the order they run. Use this file OR the small scripts; both produce the same output.
The small scripts stay the official version that the model code imports.

Read it top to bottom. Each numbered section matches one small script:

    0. Settings          config.py         every number the pipeline uses, in one place
    1. Read the labels   (explore/preprocess) load digitStruct.json
    2. Explore           explore_data.py   counts and sizes that drive our decisions (k = 5, why we crop)
    3. Preprocess        preprocess.py     crop -> resize -> labels -> train/val/test split -> save .npz
    4. Hand-off          load_data.py      make_dataset(): what the model (Person 2) trains on
    5. Checks            check_handoff.py  prove the labels and the hand-off are right
    6. Figures           make_figures.py   the charts on our slides

Run everything:          python data_pipeline/svhn_pipeline.py
Run one step:            python data_pipeline/svhn_pipeline.py --step explore
                         (steps: explore, preprocess, check, figures)

Set the SVHN_DIR environment variable if your raw data is not in data/SVHN.
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import tensorflow as tf
from PIL import Image


# =====================================================================================
# 0. Settings
# =====================================================================================
HERE = os.path.dirname(os.path.abspath(__file__))

# Folder that holds train/ and test/ (each with PNGs and digitStruct.json).
SVHN_DIR = os.environ.get("SVHN_DIR", os.path.join(HERE, "..", "data", "SVHN"))
PROCESSED_DIR = os.path.join(HERE, "..", "data", "processed")  # where step 3 saves its output
FIGURES_DIR = os.path.join(HERE, "..", "figures")              # where step 6 saves the charts

IMG_SIZE = 64        # every crop is resized to IMG_SIZE x IMG_SIZE
MARGIN = 0.30        # grow the digit box by 30% on each axis before cropping (Goodfellow et al.)
MAX_DIGITS = 5       # k: the longest number we model (step 2 shows why)
BLANK = 10           # digit-head class meaning "no digit in this position"
VAL_FRACTION = 0.10  # share of train held out for validation
SEED = 42            # same seed = same validation split for the whole team

# One softmax head per label: "length", then "digit1" ... "digit5".
HEAD_NAMES = ["length"] + [f"digit{i + 1}" for i in range(MAX_DIGITS)]


# =====================================================================================
# 1. Read the labels
# =====================================================================================
def load_struct(split):
    """
    Read digitStruct.json for 'train' or 'test'. Returns a list with one entry per image:
        {"filename": "33402.png",
         "boxes": [{"left": 35, "top": 10, "width": 7, "height": 25, "label": 1}, ...]}
    One box per digit, in reading order. SVHN stores the digit 0 as label 10.
    """
    with open(os.path.join(SVHN_DIR, split, "digitStruct.json")) as f:
        return json.load(f)


# =====================================================================================
# 2. Explore: look at the raw data before changing anything
# =====================================================================================
def explore():
    for split in ["train", "test"]:
        records = load_struct(split)
        lengths = [len(r["boxes"]) for r in records]

        # Image sizes. PIL only reads the file header here, so this is fast.
        sizes = np.array([Image.open(os.path.join(SVHN_DIR, split, r["filename"])).size
                          for r in records])  # each row is (width, height)

        # Images with a box that sticks out of the picture (negative top/left, or past the edge).
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
        print(f"width  min/median/max: {sizes[:, 0].min()} / {int(np.median(sizes[:, 0]))} / {sizes[:, 0].max()}")
        print(f"height min/median/max: {sizes[:, 1].min()} / {int(np.median(sizes[:, 1]))} / {sizes[:, 1].max()}")
        print(f"images with a box outside the image: {out_of_bounds}")
        print(f"raw label counts (10 means digit 0): {dict(sorted(labels.items()))}")
        print()
    # What this told us: train has one 6-digit image and test tops out at 5, so k = 5.
    # Widths run from 25 to 1,083 px, so we must crop and resize.
    # 181 train + 39 test images have a box past the edge, so the crop must be clipped.


# =====================================================================================
# 3. Preprocess: crop -> resize -> labels -> split -> save
# =====================================================================================
def crop_box(boxes, img_w, img_h):
    """One box around ALL digits, grown by MARGIN, clipped to the image. Returns (left, top, right, bottom)."""
    left = min(b["left"] for b in boxes)
    top = min(b["top"] for b in boxes)
    right = max(b["left"] + b["width"] for b in boxes)
    bottom = max(b["top"] + b["height"] for b in boxes)

    # Grow by MARGIN in total on each axis: half on each side.
    pad_x = (right - left) * MARGIN / 2
    pad_y = (bottom - top) * MARGIN / 2

    # Clip, because some boxes start at negative pixels or run past the edge.
    return (max(0, int(left - pad_x)), max(0, int(top - pad_y)),
            min(img_w, int(right + pad_x)), min(img_h, int(bottom + pad_y)))


def make_labels(boxes):
    """
    Length label + one label per position. Example, house number 169:
        length = 3, digits = [1, 6, 9, 10, 10]
    SVHN stores 0 as 10, so we map 10 -> 0 first.
    That frees 10 to mean "blank" for positions past the end of the number.
    """
    digits = [0 if int(b["label"]) == 10 else int(b["label"]) for b in boxes]
    padded = digits + [BLANK] * (MAX_DIGITS - len(digits))
    return len(digits), padded


def process_split(split):
    """Crop, resize and label every image in 'train' or 'test'. Returns X, y_len, y_digits, files."""
    folder = os.path.join(SVHN_DIR, split)
    X, y_len, y_digits, files, skipped = [], [], [], [], []

    for r in load_struct(split):
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


def save_split(name, X, y_len, y_digits, files):
    """
    Save one split to data/processed/<name>.npz:
        X         uint8 (N, 64, 64, 3)
        y_len     int32 (N,)     number of digits, 1..5
        y_digits  int32 (N, 5)   digit per position, 10 = blank
        files     str   (N,)     original filename, for checking mistakes later
    """
    path = os.path.join(PROCESSED_DIR, f"{name}.npz")
    np.savez(path, X=X, y_len=y_len, y_digits=y_digits, files=files)
    print(f"saved {name}: X {X.shape}, y_len {y_len.shape}, y_digits {y_digits.shape} -> {path}")


def preprocess():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Train -> shuffle the row numbers with a fixed seed, then the first 10% become validation.
    X, y_len, y_digits, files = process_split("train")
    idx = np.random.default_rng(SEED).permutation(len(X))
    n_val = int(len(X) * VAL_FRACTION)
    val, tr = idx[:n_val], idx[n_val:]
    save_split("train", X[tr], y_len[tr], y_digits[tr], files[tr])
    save_split("val", X[val], y_len[val], y_digits[val], files[val])

    # Test stays untouched until the final model.
    save_split("test", *process_split("test"))


# =====================================================================================
# 4. Hand-off: what the model trains on
# =====================================================================================
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
    """
    A tf.data pipeline of (x, y) batches:
        x  float32 (batch, 64, 64, 3), pixels scaled to [0, 1]
        y  dict of 6 int32 arrays, one per softmax head:
             "length"            1..5        -> softmax with MAX_DIGITS + 1 = 6 classes (0..5)
             "digit1".."digit5"  0..9 or 10  -> softmax with 11 classes
    The model's output layers must use these exact names so Keras matches each label to its head.
    Use sparse_categorical_crossentropy for every head (labels are integers, not one-hot).
    """
    X, y_len, y_digits, _ = load_arrays(split)
    ds = tf.data.Dataset.from_tensor_slices((X, to_label_dict(y_len, y_digits)))
    if shuffle:
        ds = ds.shuffle(len(X), seed=SEED, reshuffle_each_iteration=True)
    # Scale pixels after batching: cheaper, and X stays uint8 (4x less memory) until then.
    ds = ds.batch(batch_size).map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    return ds.prefetch(tf.data.AUTOTUNE)


def decode(y_len, y_digits):
    """Labels or predictions back to the house number string, e.g. (3, [1,6,9,10,10]) -> '169'."""
    return "".join(str(d) for d in y_digits[:y_len])


# =====================================================================================
# 5. Checks: prove the labels and the hand-off are right
# =====================================================================================
def check():
    # (a) Inside the number every position is a real digit; past the end every position is blank.
    for split in ["train", "val", "test"]:
        X, y_len, y_digits, files = load_arrays(split)
        for n, digits in zip(y_len, y_digits):
            assert all(d != BLANK for d in digits[:n]) and all(d == BLANK for d in digits[n:])
        print(f"{split}: {X.shape}, lengths 1-5 = {np.bincount(y_len, minlength=6)[1:]}, labels OK")

    # (b) The example from the assignment PDF: 33402.png is the house number 169.
    for split in ["train", "val"]:
        X, y_len, y_digits, files = load_arrays(split)
        hit = np.where(files == "33402.png")[0]
        if len(hit):
            i = hit[0]
            print(f"33402.png ({split}): y_len={y_len[i]}, y_digits={y_digits[i]}, "
                  f"decoded='{decode(y_len[i], y_digits[i])}'")

    # (c) One batch from the tf.data pipeline has the right shapes and pixel range.
    x, y = next(iter(make_dataset("train", batch_size=8, shuffle=True)))
    print("batch x:", x.shape, x.dtype, f"range [{x.numpy().min():.2f}, {x.numpy().max():.2f}]")
    print("batch y:", {k: v.shape for k, v in y.items()})

    # (d) A throwaway 6-head model trains on it for a few steps. This is NOT our model
    #     (that's Person 2's); it only proves the head names and shapes line up.
    inp = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    h = tf.keras.layers.Conv2D(16, 3, activation="relu")(inp)
    h = tf.keras.layers.GlobalAveragePooling2D()(h)
    outputs = {"length": tf.keras.layers.Dense(MAX_DIGITS + 1, activation="softmax", name="length")(h)}
    for name in HEAD_NAMES[1:]:
        outputs[name] = tf.keras.layers.Dense(11, activation="softmax", name=name)(h)
    model = tf.keras.Model(inp, outputs)
    model.compile("adam", loss={k: "sparse_categorical_crossentropy" for k in HEAD_NAMES})
    model.fit(make_dataset("train", batch_size=64).take(5), epochs=1, verbose=2)
    print("hand-off OK: 6 named heads train on make_dataset()")


# =====================================================================================
# 6. Figures for the slides -> figures/
# =====================================================================================
def make_figures():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    NAVY, BLUE, ORANGE, AQUA = "#0f2a4a", "#2a78d6", "#eb6834", "#1baf7a"
    INK, MUTED, GRAY, LINE = "#0b0b0b", "#52514e", "#b5b4ae", "#d6d5d0"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "text.color": INK,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    os.makedirs(FIGURES_DIR, exist_ok=True)

    def save(fig, name, bg="white"):
        fig.savefig(os.path.join(FIGURES_DIR, name), facecolor=bg, dpi=200,
                    bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    records = load_struct("train")
    by_name = {r["filename"]: r for r in records}

    # --- Title slide: 2 rows x 12 real crops, preferring high-contrast ones ---
    X, *_ = load_arrays("train")
    contrast = X.reshape(len(X), -1).std(axis=1)
    sharp = np.where(contrast > np.percentile(contrast, 75))[0]
    pick = np.random.default_rng(7).choice(sharp, 24, replace=False)
    fig, axes = plt.subplots(2, 12, figsize=(19.2, 3.3))
    fig.subplots_adjust(wspace=0.06, hspace=0.08, left=0, right=1, top=1, bottom=0)
    for a, i in zip(axes.flat, pick):
        a.imshow(X[i], interpolation="bilinear")
        a.axis("off")
    save(fig, "title_mosaic.png", bg=NAVY)

    # --- Digits per image (why k = 5) ---
    counts = Counter(len(r["boxes"]) for r in records)
    lengths = sorted(counts)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.bar(lengths, [counts[n] for n in lengths], width=0.62,
                  color=[GRAY if n > 5 else BLUE for n in lengths])
    for n, b in zip(lengths, bars):
        note = f"{counts[n]:,}" + ("\ndropped" if n > 5 else "")
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 300, note, ha="center", va="bottom",
                fontsize=18, color=MUTED if n > 5 else INK, fontweight="normal" if n > 5 else "bold")
    ax.set_xlabel("Digits in the house number", fontsize=18, labelpad=10)
    ax.set_xticks(lengths)
    ax.tick_params(axis="x", length=0, labelsize=18)
    ax.set_yticks([])
    ax.set_ylim(0, max(counts.values()) * 1.18)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    save(fig, "digits_per_image.png")

    # --- Label encoding: the photo of 169 -> 6 labels ---
    r = by_name["33402.png"]
    img = Image.open(os.path.join(SVHN_DIR, "train", r["filename"])).convert("RGB")
    crop = img.crop(crop_box(r["boxes"], *img.size)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    length, digits = make_labels(r["boxes"])

    fig = plt.figure(figsize=(14, 4.2))
    ax_img = fig.add_axes([0.0, 0.12, 0.2, 0.76])
    ax_img.imshow(crop, interpolation="bilinear")
    ax_img.axis("off")
    ax = fig.add_axes([0.21, 0, 0.79, 1])
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
    ax.annotate("", xy=(11, 20), xytext=(1, 20),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2.5, mutation_scale=28))
    values = [str(length)] + ["–" if d == BLANK else str(d) for d in digits]
    names = ["length"] + [f"digit {i + 1}" for i in range(MAX_DIGITS)]
    for i, (v, n) in enumerate(zip(values, names)):
        x = 14 + i * 14.2 + (2 if i else 0)
        color = ORANGE if i == 0 else BLUE
        blank = v == "–"
        ax.add_patch(FancyBboxPatch((x, 9), 11.5, 21, boxstyle="round,pad=0,rounding_size=1.6",
                                    fc="white", ec=GRAY if blank else color, lw=3))
        ax.text(x + 5.75, 19.5, v, ha="center", va="center", fontsize=46, fontweight="bold",
                color=GRAY if blank else INK)
        ax.text(x + 5.75, 34, n, ha="center", va="center", fontsize=19, color=MUTED)
    ax.text(19.75, 3.5, "6 classes", ha="center", fontsize=17, color=ORANGE)
    ax.text(16 + 14.2 * 3 + 5.75, 3.5, "11 classes each: 0–9 or blank", ha="center", fontsize=17, color=BLUE)
    save(fig, "label_encoding.png")

    # --- Crop flow: raw -> crop (+30%) -> 64x64 ---
    l, t, rr, b = crop_box(r["boxes"], *img.size)
    raw_w, raw_h = img.size
    crop_w, crop_h = rr - l, b - t
    fig = plt.figure(figsize=(16, 3.6))
    widths = [raw_w / raw_h, 0.45, crop_w / crop_h, 0.45, 1.0]  # panels at equal height, arrows between
    gs = fig.add_gridspec(1, 5, width_ratios=widths, wspace=0.05)
    a0, a1, a2, a3, a4 = (fig.add_subplot(gs[i]) for i in range(5))
    a0.imshow(img, interpolation="bilinear")
    for bx in r["boxes"]:
        a0.add_patch(Rectangle((bx["left"], bx["top"]), bx["width"], bx["height"], fill=False, ec=BLUE, lw=2.5))
    a0.add_patch(Rectangle((l, t), crop_w, crop_h, fill=False, ec=ORANGE, lw=3, ls="--"))
    a2.imshow(img.crop((l, t, rr, b)), interpolation="bilinear")
    a4.imshow(crop, interpolation="nearest")
    for a, c in [(a0, f"Raw image, {raw_w}×{raw_h} px"), (a2, "Crop: all digits + 30%"), (a4, "Resize: 64×64")]:
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_color(LINE)
        a.set_xlabel(c, fontsize=20, color=INK, labelpad=12)
    for a in (a1, a3):
        a.axis("off")
        a.annotate("", xy=(0.95, 0.5), xytext=(0.05, 0.5), xycoords="axes fraction",
                   arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2.5, mutation_scale=28))
    save(fig, "pipeline_flow.png")

    # --- Train / validation / test split ---
    parts = [("Train", len(load_arrays("train")[0]), BLUE),
             ("Validation", len(load_arrays("val")[0]), ORANGE),
             ("Test", len(load_arrays("test")[0]), AQUA)]
    total = sum(n for _, n, _ in parts)
    fig, ax = plt.subplots(figsize=(14, 1.9))
    x, gap = 0, total * 0.004
    for name, n, c in parts:
        ax.barh(0, n - gap, left=x, height=0.5, color=c)
        # Test is labeled from its right end so it never collides with the narrow validation slice.
        tx, ha = (x + n - gap, "right") if name == "Test" else (x, "left")
        ax.text(tx, 0.48, name, ha=ha, va="bottom", fontsize=19, color=MUTED)
        ax.text(tx, -0.4, f"{n:,}", ha=ha, va="top", fontsize=24, fontweight="bold", color=INK)
        x += n
    ax.set_xlim(0, total); ax.set_ylim(-1.05, 1.0); ax.axis("off")
    save(fig, "split_bar.png")

    print("saved figures to", os.path.abspath(FIGURES_DIR))


# =====================================================================================
# Run
# =====================================================================================
STEPS = {"explore": explore, "preprocess": preprocess, "check": check, "figures": make_figures}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVHN data pipeline")
    parser.add_argument("--step", choices=list(STEPS), help="run one step; leave out to run all of them")
    args = parser.parse_args()
    for name in ([args.step] if args.step else STEPS):
        print(f"\n##### {name} #####")
        STEPS[name]()
