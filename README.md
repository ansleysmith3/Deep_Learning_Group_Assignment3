# SVHN Multi-Digit Number Recognition — BZAN 554 Group Assignment 3

Recognize full house numbers (1–5 digits) from the [SVHN](http://ufldl.stanford.edu/housenumbers/)
(Street View House Numbers) dataset using a multi-output CNN, following
[Goodfellow et al. (2013)](https://arxiv.org/abs/1312.6082): one shared convolutional
backbone feeding `k+1` softmax heads (a length head + one head per digit position).

**Deliverable:** a presentation only — no code is submitted. This repo is the team's working
space; see each person's README for the detail behind their slides.

**Status: done.** All three sections are merged into `main`; the final model, test-set
result, and error analysis below are the real, final numbers — not placeholders.

## Team & role split

| Person | Role | Branch | README |
|---|---|---|---|
| Ansley | Data pipeline — raw PNGs + `digitStruct.json` → model-ready arrays | `ansley-data` | [`data_pipeline/README.md`](data_pipeline/README.md) |
| Ximin | Model & training — the multi-output CNN + baseline | `ximin-model` | [`README_model_baseline-v2.md`](README_model_baseline-v2.md) |
| Becca | Experiments, final evaluation, error analysis, deck | `Rebecca` | [`Rebecca/README_person3.md`](Rebecca/README_person3.md) |

**Branch status:** `main` now has everything — `ansley-data`, `ximin-model`, and `Rebecca`
were merged into it. The per-person branches above still exist for history/authorship, but
`main` is the branch to check out for the finished pipeline.

## Repo layout

```
data_pipeline/          Person 1 (Ansley) — preprocessing, config, the train/val/test hand-off
  config.py              IMG_SIZE=64, MAX_DIGITS=5, BLANK=10, VAL_FRACTION=0.10, SEED=42
  preprocess.py          raw PNGs + digitStruct.json -> data/processed/{train,val,test}.npz
  load_data.py           make_dataset(split) -> the tf.data pipeline the model trains on
  check_handoff.py       sanity checks + trains a throwaway model for 5 steps
  explore_data.py        dataset stats (digit-length distribution, k=5, etc.)
  make_figures.py        writes figures/ for the data slides

model_baseline_ximin_v2.py   Person 2 (Ximin) — the multi-output CNN + training loop
evaluate_baseline.py          loads the saved checkpoint and reports val whole-number accuracy
artifacts/                    baseline_best.keras (shared checkpoint), curves, history

Rebecca/                 Person 3 (Becca) — experiments, final eval, error analysis, deck
  metrics.py              whole-number + per-digit accuracy, used everywhere below
  run_experiment.py       change one knob, train, log the result
  evaluate_test_final.py  the single final run on the held-out test set
  error_analysis.py       groups wrong predictions, saves an example grid for the deck
  README_person3.md       run order + exam-prep notes
  slide_outline.md        slide-by-slide template
  slide_content_draft.md  the outline filled in with real numbers (ready to paste into Canva)

experiments/              run_experiment.py's output — one folder per run + a shared log
  experiment_log.csv       one row per run: change, val whole-number accuracy, settings, notes
  final_test_results.json  the one-time final test-set numbers
  error_analysis/          the saved error-example grid image(s)

figures/                  slide figures from make_figures.py (data-side)
data/                     NOT in git (.gitignore) — everyone generates this locally, see below
```

## Setup — reproducing this from scratch

1. **Get the code**: clone/pull this repo and check out `main` — everything is merged there.
2. **Install dependencies**: `tensorflow >= 2.16`, `numpy`, `matplotlib`, `pillow`.
   > TensorFlow ≥2.11 has no GPU support on native Windows — expect CPU-only training
   > (~40 minutes per full run) unless you use Colab or WSL2.
3. **Get the data**: download [`SVHN.zip`](http://ballings.co/SVHN.zip) and extract it with
   `tar` or 7-Zip/WinRAR — **not** Windows Explorer's built-in unzip, which silently drops
   files. You want `train/` (33,402 PNGs) and `test/` (13,068 PNGs), each with its own
   `digitStruct.json`. By default the pipeline looks for these at `data/SVHN/{train,test}`;
   if your data lives somewhere else, set the `SVHN_DIR` environment variable instead of
   moving tens of thousands of files around.
4. **Preprocess once**:
   ```
   python data_pipeline/preprocess.py     # writes data/processed/{train,val,test}.npz (~600MB)
   python data_pipeline/check_handoff.py  # sanity-checks labels + shapes
   ```
5. **Get the baseline checkpoint**: `artifacts/baseline_best.keras` is tracked in git on
   `main` (an explicit exception to `.gitignore`'s `*.keras` rule), so a plain checkout
   gives it to you — no need to retrain the ~40-minute baseline yourself.
6. **Run experiments / final eval / error analysis** — see
   [`Rebecca/README_person3.md`](Rebecca/README_person3.md) for the exact commands.

## Key design decisions (full detail in `data_pipeline/README.md`)

- **k=5 digits → 6 softmax heads** (`length`: 6 classes 0–5; `digit1..digit5`: 11 classes
  each, 0–9 plus `10` = "no digit here"). One 6-digit training image was dropped — it can't
  teach a 6th head anything, and the test set tops out at 5 digits.
- **SVHN's raw label 10 (meaning digit '0') is remapped to 0**, freeing class 10 to mean
  "blank" for digit positions beyond a number's actual length.
- **Crop to the union of all digit boxes, grown 30%, then resize to 64×64** — matches
  Goodfellow et al.'s margin and keeps digit edges from being cut off.
- **90/10 train/val split (seed 42)**; the **test set is touched exactly once**, at the very
  end, per the assignment's own rule.
- **Validation has zero 5-digit examples** (train only has 9) — val accuracy can't actually
  verify 5-digit generalization; a real caveat to mention in the deck.

## Results

| Run | Change | Val whole-number accuracy | Notes |
|---|---|---|---|
| `baseline` | — | 76.20% (2,545/3,340) | Ximin's baseline; best checkpoint at epoch 5, overfits after |
| `augmentation` | rotation/translation/contrast | 80.03% (2,673/3,340) | |
| `lr_sched` | ReduceLROnPlateau + EarlyStopping | 80.12% (2,676/3,340) | Stopped at epoch 11, ~26 min vs. ~42 min |
| **`combined`** | **all three together** | **82.72% (2,763/3,340)** | **Final model** |

**Final test-set result (run once): 82.66% whole-number accuracy (10,802/13,068).** Per-head
accuracy: length 96.55%, digit1 91.87%, digit2 91.09%, digit3 95.96%, digit4 99.46%,
digit5 99.98% — the trailing heads look inflated mainly because most numbers are short.

Full experiment log: [`experiments/experiment_log.csv`](experiments/experiment_log.csv).
Full write-up with error analysis: [`Rebecca/slide_content_draft.md`](Rebecca/slide_content_draft.md).

## Exam prep

Everyone should be able to explain all three parts, not just their own section — each
person's README has an exam-prep section at the bottom (data decisions in
`data_pipeline/README.md`, architecture/inference in `README_model_baseline-v2.md`, metrics
and ablations in `Rebecca/README_person3.md`).
