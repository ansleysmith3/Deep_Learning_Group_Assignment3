# Person 3 slide content — ready to paste into Canva

Real numbers from `experiments/experiment_log.csv`, `experiments/final_test_results.json`,
and `experiments/error_analysis/error_examples_test.png`.

---

## Slide 1 — Final Result

**Final test whole-number accuracy: 82.66% (10,802 / 13,068)**

- Up from a 76.20% baseline — a +6.5 point improvement
- Val accuracy was 82.72% — nearly identical to test, so the model generalizes well and
  isn't overfit to validation

Per-head test accuracy:

| Head | Accuracy |
|---|---|
| length | 96.55% |
| digit1 | 91.87% |
| digit2 | 91.09% |
| digit3 | 95.96% |
| digit4 | 99.46% |
| digit5 | 99.98% |

*Note: digit4/digit5 look near-perfect mainly because most house numbers are 1–3 digits, so
those positions are usually blank. Whole-number accuracy (82.66%) is the real headline —
per-head numbers can be misleading on their own.*

---

## Slide 2 — What We Tried

| Run | Change | Val whole-number acc. | Epochs | Train time |
|---|---|---|---|---|
| Baseline | — | 76.20% | 15 | 42.4 min |
| Augmentation | Random rotation/translation/contrast | 80.03% | 15 | 43.2 min |
| LR schedule + early stop | ReduceLROnPlateau + EarlyStopping(patience=4) | 80.12% | 11 (stopped early) | 26.0 min |
| **Combined (final)** | **All three together** | **82.72%** | 15 | 39.9 min |

**What worked:** Both augmentation and the LR schedule/early-stopping combo independently
beat the baseline by ~4 points — and when stacked together, the gains were roughly
additive (+6.5 points total over baseline), suggesting they fix different problems:
augmentation adds visual variety the model hadn't seen, while early stopping/LR decay
directly countered the overfitting the baseline showed after epoch 5.

**What we'd try next if we had more time:** more conv capacity (extra conv block),
L2 regularization, and a longer augmentation sweep — flagged in `run_experiment.py` as
knobs that are ready to go but we ran out of runway before presenting.

---

## Slide 3 — Error Analysis

*(Insert `experiments/error_analysis/error_examples_test.png` here)*

Of 13,068 test images, 2,266 were wrong (17.3%):

| Failure type | Count | Share of errors |
|---|---|---|
| Blurry / low-contrast crop | 778 | 34% |
| Other / single-digit swap | 704 | 31% |
| Length head wrong | 451 | 20% |
| Cropped badly (digit cut off at edge) | 241 | 11% |
| Long number (4+ digits) | 92 | 4% |

**Dominant failure mode:** image quality, not model capacity. The largest bucket (blurry)
is images that would be hard for a person to read too. The "other" bucket has several
near-misses — single-digit swaps like 81→21, 288→286, 260→160 — where the model got the
general shape right but confused visually similar digits.

---

## Slide 4 — Challenges

- **Compute:** every full training run took ~40+ minutes on CPU (no GPU support on native
  Windows for TensorFlow ≥2.11); this capped us at a handful of experiments rather than a
  broad sweep
- **Data imbalance:** long numbers (4-5 digits) are rare — validation data has zero 5-digit
  examples, so we can't fully verify generalization to the longest numbers
- **Metric interpretation:** per-head accuracy alone is misleading (trailing heads look
  inflated); had to be deliberate about reporting whole-number accuracy as the real measure
- **Team coordination:** data → model → experiments is a strict pipeline with one hard
  hand-off point; getting the trained checkpoint shared between teammates (it's a 28MB
  binary, excluded by default `.gitignore`) needed an explicit fix mid-stream

---

## Exam-prep notes (say these out loud, no slide needed)

- Whole-number accuracy only counts a prediction correct if every digit AND the length are
  right — that's why it's lower than any single per-head number
- BLANK=10 exists because raw digitStruct.json already uses label 10 for digit '0';
  preprocessing remaps raw 10→0, freeing class 10 to mean "no digit here"
- Inference: argmax(length) picks how many of the 5 digit heads to read; decode() joins
  those digits into a string
- Why k+1 heads instead of one big multi-class head: a 5-digit number has 10^5 = 100,000
  combinations; decomposing into 1 length head (6 classes) + 5 digit heads (11 classes
  each) = 61 output neurons instead, and lets every position share the same conv features
