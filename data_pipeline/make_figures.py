"""
Figures for the slides -> figures/.
  title_mosaic.png       real 64x64 crops, for the title slide (navy background)
  digits_per_image.png   how many digits each training image has (why k = 5)
  label_encoding.png     how one image becomes 6 labels, one per softmax head
  pipeline_flow.png      raw image -> crop (+30%) -> 64x64
  split_bar.png          train / val / test sizes

Needs data/processed/ (run preprocess.py first).
Run:  python data_pipeline/make_figures.py
"""
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

from config import IMG_SIZE, SVHN_DIR
from load_data import load_arrays
from preprocess import crop_box, make_labels

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
NAVY, BLUE, ORANGE, AQUA = "#0f2a4a", "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRAY, LINE = "#0b0b0b", "#52514e", "#b5b4ae", "#d6d5d0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
os.makedirs(OUT, exist_ok=True)


def save(fig, name, bg="white"):
    fig.savefig(os.path.join(OUT, name), facecolor=bg, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


with open(os.path.join(SVHN_DIR, "train", "digitStruct.json")) as f:
    records = json.load(f)
by_name = {r["filename"]: r for r in records}

# --- Title mosaic: 2 rows x 12 real crops ---------------------------------------
X, *_ = load_arrays("train")
# Prefer high-contrast crops so the digits read at a glance.
contrast = X.reshape(len(X), -1).std(axis=1)
sharp = np.where(contrast > np.percentile(contrast, 75))[0]
pick = np.random.default_rng(7).choice(sharp, 24, replace=False)
fig, axes = plt.subplots(2, 12, figsize=(19.2, 3.3))
fig.subplots_adjust(wspace=0.06, hspace=0.08, left=0, right=1, top=1, bottom=0)
for a, i in zip(axes.flat, pick):
    a.imshow(X[i], interpolation="bilinear")
    a.axis("off")
save(fig, "title_mosaic.png", bg=NAVY)

# --- Digits per image ------------------------------------------------------------
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

# --- Label encoding: 169 -> 6 labels ----------------------------------------------
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
ax.annotate("", xy=(11, 20), xytext=(1, 20), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2.5,
                                                           mutation_scale=28))
values = [str(length)] + ["–" if d == 10 else str(d) for d in digits]
names = ["length"] + [f"digit {i + 1}" for i in range(5)]
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

# --- Pipeline flow: raw -> crop -> 64x64 ------------------------------------------
l, t, rr, b = crop_box(r["boxes"], *img.size)
raw_w, raw_h = img.size
crop_w, crop_h = rr - l, b - t
fig = plt.figure(figsize=(16, 3.6))
# panel widths proportional to aspect ratio at equal height; arrows between
widths = [raw_w / raw_h, 0.45, crop_w / crop_h, 0.45, 1.0]
gs = fig.add_gridspec(1, 5, width_ratios=widths, wspace=0.05)
a0, a1, a2, a3, a4 = (fig.add_subplot(gs[i]) for i in range(5))
a0.imshow(img, interpolation="bilinear")
for bx in r["boxes"]:
    a0.add_patch(Rectangle((bx["left"], bx["top"]), bx["width"], bx["height"], fill=False, ec=BLUE, lw=2.5))
a0.add_patch(Rectangle((l, t), crop_w, crop_h, fill=False, ec=ORANGE, lw=3, ls="--"))
a2.imshow(img.crop((l, t, rr, b)), interpolation="bilinear")
a4.imshow(crop, interpolation="nearest")
captions = [(a0, f"Raw image, {raw_w}×{raw_h} px"), (a2, "Crop: all digits + 30%"),
            (a4, "Resize: 64×64")]
for a, c in captions:
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_color(LINE)
    a.set_xlabel(c, fontsize=20, color=INK, labelpad=12)
for a in (a1, a3):
    a.axis("off")
    a.annotate("", xy=(0.95, 0.5), xytext=(0.05, 0.5), xycoords="axes fraction",
               arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2.5, mutation_scale=28))
save(fig, "pipeline_flow.png")

# --- Split bar ------------------------------------------------------------------
parts = [("Train", 30061, BLUE), ("Validation", 3340, ORANGE), ("Test", 13068, AQUA)]
total = sum(n for _, n, _ in parts)
fig, ax = plt.subplots(figsize=(14, 1.9))
x = 0
gap = total * 0.004
for name, n, c in parts:
    ax.barh(0, n - gap, left=x, height=0.5, color=c)
    # Test is labeled from its right end so it never collides with the narrow validation slice.
    tx, ha = (x + n - gap, "right") if name == "Test" else (x, "left")
    ax.text(tx, 0.48, name, ha=ha, va="bottom", fontsize=19, color=MUTED)
    ax.text(tx, -0.4, f"{n:,}", ha=ha, va="top", fontsize=24, fontweight="bold", color=INK)
    x += n
ax.set_xlim(0, total); ax.set_ylim(-1.05, 1.0); ax.axis("off")
save(fig, "split_bar.png")

print("saved figures to", os.path.abspath(OUT))
