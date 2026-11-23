# Person 3 — Experiments, Evaluation & Deck (Becca)

Picks up where `README_model_baseline-v2.md` (Ximin, Person 2) leaves off. Baseline: **76.20%
whole-number val accuracy** (2,545/3,340), best checkpoint `artifacts/baseline_best.keras`
at epoch 5.

All four scripts below were smoke-tested end-to-end on fake data in a sandbox (`--fast_debug`
path) before being committed, so the code runs — what it hasn't seen is the real SVHN data.

## What's new in this branch (`Rebecca`, based on `ximin-model`)

| File | Purpose |
|---|---|
| `metrics.py` | `evaluate_model(model, dataset)` → whole-number accuracy + per-head accuracy, in one place so every run reports numbers the same way. |
| `run_experiment.py` | Change ONE knob, train, log the result to `experiments/experiment_log.csv`. Seeds the log with Ximin's baseline row automatically. |
| `evaluate_test_final.py` | The single final run on the held-out test set. Run this exactly once, on whichever model wins. |
| `error_analysis.py` | Groups wrong predictions (length_wrong / long_number / cropped_badly / blurry / other) and saves an example grid image for the deck. |

## Run order (once you have `data/SVHN/{train,test}` + have run `data_pipeline/preprocess.py`)

1. **Reproduce or fetch the baseline**, if `artifacts/baseline_best.keras` isn't already on
   your machine (it's gitignored — ask Ximin to share the `.keras` file directly, e.g. via
   the group Drive, instead of retraining 42 min if you can avoid it):
   ```bash
   python model_baseline_ximin_v2.py   # run from the repo root; only if you don't have baseline_best.keras
   ```
2. **Run 2–3 experiments, one knob each** (per the timeline: "no time for a long sweep"):
   ```bash
   python Rebecca/run_experiment.py --name augmentation --change "data augmentation" --augment
   python Rebecca/run_experiment.py --name dropout05 --change "dropout 0.3->0.5" --dropout 0.5
   python Rebecca/run_experiment.py --name lr_sched --change "LR schedule + early stop" --lr_schedule --early_stopping
   ```
   Check `experiments/experiment_log.csv` after each — it's the "what worked / what didn't" slide.
3. **Pick the best `val_whole_number_accuracy` row**, then run the test set ONCE:
   ```bash
   python Rebecca/evaluate_test_final.py --model experiments/<winning_run>/model_best.keras
   ```
4. **Error analysis** on that same model:
   ```bash
   python Rebecca/error_analysis.py --model experiments/<winning_run>/model_best.keras --split test
   ```
5. **Build the deck** (Canva link from the assignment doc) using `slide_outline.md` below,
   pulling in Ansley's and Ximin's slides, then rehearse and submit via the Dropbox link.

## Time budget (today is Mon Nov 23, presenting Tue Nov 24)

A full baseline run took Ximin ~42 min on CPU. Each experiment here is a full retrain, so
budget similarly per run unless you're on a GPU (Colab/ISAAC) — in which case it should be
much faster. With ~2–3 experiments planned and an evening merge/rehearsal slot, start the
first run now rather than after the code walkthrough.

## Slide outline (`slide_outline.md` has the full version)

Final test whole-number + per-digit accuracy → experiment log highlights → error examples
(from the saved grid image) → challenges.

## Running these scripts

All four scripts live in `Rebecca/` but import `data_pipeline/` and `model_baseline_ximin_v2.py` from the repo root (a small `sys.path` shim at the top of each file handles this). Run them **from the repo root**, e.g. `python Rebecca/run_experiment.py --name ...` — not from inside `Rebecca/`. Output (`experiments/`) also lands at the repo root, next to `artifacts/`.
