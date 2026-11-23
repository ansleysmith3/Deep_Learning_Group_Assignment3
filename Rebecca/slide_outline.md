# Person 3 Slides — outline to fill in once real numbers exist

Matches line 40 of the work-split doc: "final test accuracy, what worked and what didn't
(from the log), error examples, and challenges."

## Slide 1 — Final result
- Headline: **FINAL test whole-number accuracy: __.__%** (from `experiments/final_test_results.json`)
- One line comparing it to the 76.20% baseline
- Per-head accuracy table (length, digit1..digit5) — note trailing heads (digit4/digit5) look
  inflated because most numbers are short; the whole-number number is the real one

## Slide 2 — What we tried (from experiment_log.csv)
- Small table: run name | change | val whole-number accuracy | notes
- One sentence on what helped, one on what didn't (or made it worse) and your best guess why
- Name the winning config going into the final test run

## Slide 3 — Error analysis
- The saved grid from `error_analysis.py` (`experiments/error_analysis/error_examples_test.png`)
- Counts per group: length_wrong / long_number / cropped_badly / blurry / other
- 1-2 sentences on the dominant failure mode

## Slide 4 — Challenges
- Data/compute: dataset size, CPU training time (~42 min/run), GPU need
- Modeling: val.npz has zero 5-digit numbers (can't validate long-number generalization)
- Anything team-process related (hand-off timing, merging branches, etc.)

## Exam-prep notes (you should be able to explain, out loud, without the deck)
- Why whole-number accuracy is the real metric and per-head accuracy can mislead
- Why BLANK=10 and label 10 (raw) -> 0 remapping exists
- How predict_house_numbers() decodes: argmax(length) picks how many of the 5 digit heads to read
- What each experiment changed and why you expected it to help
