"""
Figures for the data slides -> figures/.
  digits_per_image.png   how many digits each training image has (why k = 5)
  pipeline_examples.png  raw image + boxes -> 64x64 crop, with the labels we build

Run:  python data_pipeline/make_figures.py
"""
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from config import IMG_SIZE, SVHN_DIR
from preprocess import crop_box, make_labels

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#a8a7a2", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(SVHN_DIR, "train", "digitStruct.json")) as f:
    records = json.load(f)
by_name = {r["filename"]: r for r in records}

# --- 1. Digits per image -------------------------------------------------------
counts = Counter(len(r["boxes"]) for r in records)
lengths = sorted(counts)
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
bars = ax.bar(lengths, [counts[n] for n in lengths], width=0.6,
              color=[GRAY if n > 5 else BLUE for n in lengths])
for n, b in zip(lengths, bars):
    note = f"{counts[n]:,}" + ("\n(dropped)" if n > 5 else "")
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 250, note, ha="center", va="bottom",
            fontsize=12, color=INK)
ax.set_xlabel("Digits in the house number")
ax.set_ylabel("Training images")
ax.set_title("Most numbers have 2 digits; the longest we model is 5", loc="left", fontsize=14)
ax.set_ylim(0, max(counts.values()) * 1.15)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color("#d6d5d0")
ax.yaxis.grid(True, color="#ecebe7")
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "digits_per_image.png"), facecolor="white")
plt.close(fig)

# --- 2. Pipeline examples ------------------------------------------------------
examples = ["33402.png", "177.png"]
fig, axes = plt.subplots(len(examples), 2, figsize=(9, 3.2 * len(examples)), dpi=200,
                         gridspec_kw={"width_ratios": [2.2, 1]})
for row, name in zip(axes, examples):
    boxes = by_name[name]["boxes"]
    img = Image.open(os.path.join(SVHN_DIR, "train", name)).convert("RGB")
    l, t, r, b = crop_box(boxes, *img.size)
    length, digits = make_labels(boxes)

    row[0].imshow(img)
    for bx in boxes:
        row[0].add_patch(Rectangle((bx["left"], bx["top"]), bx["width"], bx["height"],
                                   fill=False, ec=BLUE, lw=2))
    row[0].add_patch(Rectangle((l, t), r - l, b - t, fill=False, ec=ORANGE, lw=2.5, ls="--"))
    row[0].set_title(f"{name}  ({img.size[0]}x{img.size[1]})", fontsize=12, loc="left", color=MUTED)

    row[1].imshow(img.crop((l, t, r, b)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    shown = ", ".join("_" if d == 10 else str(d) for d in digits)
    row[1].set_title(f"length {length}\ndigits [{shown}]", fontsize=12, loc="left")
    for a in row:
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_color("#d6d5d0")

axes[0][0].legend(handles=[Rectangle((0, 0), 1, 1, fill=False, ec=BLUE, lw=2),
                           Rectangle((0, 0), 1, 1, fill=False, ec=ORANGE, lw=2.5, ls="--")],
                  labels=["digit box (from JSON)", "crop: all digits + 30%"],
                  loc="upper left", bbox_to_anchor=(0, -0.02), ncol=2, frameon=False, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "pipeline_examples.png"), facecolor="white")
plt.close(fig)
print("saved figures to", os.path.abspath(OUT))
