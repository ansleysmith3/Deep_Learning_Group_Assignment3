# Baseline Multi-Output CNN (`model_baseline_ximin_v2.py` & `evaluate_baseline.py`)

**Role:** Person 2 — Model & Training (Ximin)  
**Course:** BZAN 554 • Group Assignment 3: Multi-Digit Number Recognition (SVHN)  
**Architecture Reference:** Goodfellow et al. (2013), *Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks*

---

## 1. Overview & Model Architecture

This module implements a **multi-output Convolutional Neural Network (CNN)** using the Keras Functional API to recognize variable-length house numbers (`k = 5` max digits) from preprocessed `64 × 64 × 3` SVHN images.

* **Shared Convolutional Feature Extractor:** Three VGG-style convolutional blocks (`conv_filters = [32, 64, 128]`), each containing two `3 × 3` `Conv2D(..., padding="same", activation="relu")` layers followed by `MaxPooling2D(pool_size=2)` and `BatchNormalization()`.
* **Shared Dense Representation:** `Flatten()` (`8,192` features) → `Dense(256, activation="relu")` → `Dropout(0.3)`.
* **k + 1 Softmax Output Heads (`6` total):**
  * `length`: `Dense(6, activation="softmax", name="length")` — predicts digit count (`1..5`; class `0` unused).
  * `digit1` … `digit5`: `Dense(11, activation="softmax", name="digit{i}")` — predicts digit `0..9` or `10` (`BLANK` / no digit).
* **Loss & Optimization:** Compiled with `Adam(learning_rate=1e-3)` and `sparse_categorical_crossentropy` across all 6 named heads (Keras sums the 6 head losses into a single scalar training objective).

---

## 2. Environment & Repository Layout

### Required Python Packages
Ensure your project virtual environment (`.venv` in the parent `BZAN_554` directory) is active before running any scripts:
```bash
source ../../.venv/Scripts/activate
```
Required dependencies: `tensorflow >= 2.16`, `numpy`, `matplotlib`, `pillow`.

### Required Files & Folder Structure
```text
Deep_Learning_Group_Assignment3/
├── data/
│   ├── SVHN/                     # Extracted from SVHN.zip (using 7-Zip / unzip, NOT Windows Explorer)
│   │   ├── train/                # 33,402 PNGs + digitStruct.json
│   │   └── test/                 # 13,068 PNGs + digitStruct.json
│   └── processed/                # Generated once via python data_pipeline/preprocess.py
│       ├── train.npz             # 30,061 images (90% split, seed=42)
│       ├── val.npz               # 3,340 images (10% split, seed=42)
│       └── test.npz              # 13,068 images (held out until final evaluation)
├── data_pipeline/
│   ├── config.py                 # MAX_DIGITS=5, BLANK=10, IMG_SIZE=64
│   ├── preprocess.py             # Crops with 30% margin, remaps raw 10 -> 0, pads with 10
│   ├── load_data.py              # Provides make_dataset(), HEAD_NAMES, and decode()
│   └── check_handoff.py          # Verifies dataset shapes, labels, and 6-head model compatibility
├── model_baseline_ximin_v2.py    # Baseline CNN definition, training loop, and curve plotting
├── evaluate_baseline.py          # Loads baseline_best.keras and computes full val whole-number accuracy
└── artifacts/                    # Auto-created output directory
    ├── baseline_best.keras       # Lowest val_loss model checkpoint (saved at Epoch 5)
    ├── baseline_curves.png       # Two-panel training/validation loss and per-head accuracy plot
    └── baseline_history.json     # Per-epoch loss and accuracy metrics across all 6 heads
```

---

## 3. Step-by-Step Execution Guide

### Step 1: One-Time Data Extraction & Preprocessing
If you have not yet generated `data/processed/*.npz` on your machine:
1. Extract `SVHN.zip` into `data/` using **7-Zip** or `unzip` (do not use built-in Windows Explorer extraction, which silently drops files).
2. Run Person 1's preprocessing and hand-off verification scripts:
   ```bash
   python data_pipeline/preprocess.py
   python data_pipeline/check_handoff.py
   ```

### Step 2: Smoke Test (`fast_debug = True`) vs. Full Training (`fast_debug = False`)
In `model_baseline_ximin_v2.py`, toggle `CONFIG["fast_debug"]`:
* **CPU Smoke Test (`"fast_debug": True`):** Trains on 20 batches (`1,280` images) for 3 epochs (~19s on CPU) to verify shapes, checkpointing, and plotting end-to-end.
* **Full Training Run (`"fast_debug": False`):** Trains on all `30,061` training images (`470` batches/epoch) and validates on all `3,340` validation images (`53` batches/epoch) for 15 epochs:
  ```bash
  python model_baseline_ximin_v2.py
  ```

### Step 3: Evaluate Saved Checkpoint (`evaluate_baseline.py`)
To compute exact whole-number validation accuracy without retraining, load `artifacts/baseline_best.keras`:
```bash
python evaluate_baseline.py
```
*(Note: When calling `predict_house_numbers(model, val_ds, num_batches=-1)`, always pass `-1` rather than `None` so TensorFlow `dataset.take(-1)` iterates over all batches cleanly.)*

---

## 4. Baseline Results & Diagnostic Summary

| Metric / Detail | Baseline Result |
| :--- | :--- |
| **Full Validation Whole-Number Accuracy** | **`76.20%` (`2,545 / 3,340` exact matches)** |
| **Per-Head Val Accuracy (Best Epoch)** | `length` ~95%, `digit1` ~90%, `digit2` ~88%, `digit3` ~92%, `digit4` ~98%, `digit5` 100% |
| **Training Runtime (Local CPU)** | `2,546.3s` total (`169.8s` / ~2.8 min per epoch across 15 epochs) |
| **Best Checkpoint Epoch** | **Epoch 5** (`val_loss ≈ 1.35`; training loss continued dropping `4.5 → 0.25` afterward) |
| **Test Set Status** | **Untouched** (`13,068` test images reserved for Person 3's single final run) |

### Key Diagnostic Observations
1. **Overfitting After Epoch 5:** Training loss drops steadily across all 15 epochs, whereas validation loss bottoms out at Epoch 5 and climbs upward afterward. `ModelCheckpoint(monitor="val_loss", save_best_only=True)` automatically preserved the Epoch 5 weights.
2. **Trailing Head Accuracy Inflation:** `digit4` (~98%) and `digit5` (100%) appear near-perfect because most SVHN house numbers have 1–3 digits, padding trailing positions with `BLANK = 10`. Whole-number accuracy (`76.20%`) is the true headline metric.

---

## 5. Hand-Off Notes & Tuning Knobs for Person 3

To push whole-number validation accuracy beyond **76.20%**, test these modifications one at a time in `CONFIG` / `build_svhn_model()`:
* **Data Augmentation:** Add small `RandomRotation(0.05)`, `RandomTranslation(0.05, 0.05)`, and `RandomContrast(0.1)` layers after `image_input` (never use horizontal flips, which invert digit meanings).
* **Callbacks:** Add `EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)` and `ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)`.
* **Regularization:** Increase `dropout_rate` from `0.3` to `0.4`–`0.5` or add `kernel_regularizer=tf.keras.regularizers.l2(1e-4)` to combat post-Epoch 5 overfitting.
* **Validation Caveat:** `val.npz` contains **zero 5-digit numbers** (the dataset only has 9 in train and 2 in test), so validation accuracy does not measure 5-digit generalization.

---

## 6. Exam Prep & Code Walkthrough Reference

1. **Why `k + 1` Heads Instead of 1 Multi-Class Head?**  
   A 5-digit number has $10^5 = 100,000$ combinations. Decomposing into 1 length head (6 classes) + 5 digit heads (11 classes each) shrinks the output layer to **61 neurons** and allows all positions to share convolutional features.
2. **Why `BLANK = 10` Is Valid (`0..10` = 11 Classes):**  
   Raw `digitStruct.json` encodes digit `'0'` as `10`. `preprocess.py` remaps raw `10 → 0`, freeing class index `10` to represent `"no digit"` (`BLANK`) so `sparse_categorical_crossentropy` receives contiguous indices `0..10` and `decode()` prints `str(0)` as `"0"`.
3. **Why Output Layers Must Be Named:**  
   `load_data.py` yields `y` as a dictionary (`{"length": ..., "digit1": ..., ..., "digit5": ...}`). Naming each Keras `Dense` head identically lets Keras match each target tensor to its loss function automatically.
4. **How Inference Decoding Works:**  
   At test time, we compute $\hat{L} = rg\max(	ext{length})$, take $rg\max$ over only the first $\hat{L}$ digit heads (`digit1` … `digit_L̂`), and concatenate them via `load_data.decode()`.
