# Data pipeline (Person 1)

Turns the raw SVHN images into 64×64 arrays with one label per softmax output.

## Run it

1. Download [SVHN.zip](http://ballings.co/SVHN.zip). Windows Explorer's unzip drops most of `train/`, so extract with `tar` instead:
   ```
   tar -xf SVHN.zip -C data --exclude "__MACOSX/*"
   ```
   You should end up with `data/SVHN/train` (33,402 PNGs + `digitStruct.json`) and `data/SVHN/test` (13,068 PNGs + `digitStruct.json`). If your data lives somewhere else, set the `SVHN_DIR` environment variable.
2. `python data_pipeline/explore_data.py`: prints dataset stats (about 1 minute).
3. `python data_pipeline/preprocess.py`: writes `data/processed/{train,val,test}.npz` (a few minutes, about 600 MB).
4. `python data_pipeline/check_handoff.py`: checks the labels and trains a throwaway 6-output model for 5 steps.
5. `python data_pipeline/make_figures.py`: writes the slide figures to `figures/`.

`data/` is in `.gitignore`. Everyone runs step 3 on their own machine.

## The hand-off (for the model)

```python
from data_pipeline.load_data import make_dataset
train_ds = make_dataset("train", batch_size=64, shuffle=True)   # 30,061 images
val_ds   = make_dataset("val",   batch_size=64)                 #  3,340 images
test_ds  = make_dataset("test",  batch_size=64)                 # 13,068 images, final run only
```

| | Shape | Values |
|---|---|---|
| `x` | (batch, 64, 64, 3) float32 | pixels scaled to 0–1 |
| `y["length"]` | (batch,) int32 | 1–5. Use a softmax with **6** classes (0–5); class 0 is never used. |
| `y["digit1"]` … `y["digit5"]` | (batch,) int32 | 0–9, or **10 = no digit here**. Use a softmax with **11** classes. |

- The model's output layers must be **named** `length`, `digit1`, …, `digit5`, so Keras matches each label to its head.
- Use `sparse_categorical_crossentropy` for every head, because the labels are integers, not one-hot.
- `load_data.decode(length, digits)` turns labels or predictions back into a string, e.g. `(3, [1,6,9,10,10]) -> "169"`.

## Decisions and why

| Decision | Why |
|---|---|
| **k = 5**, so 6 softmax heads (1 length + 5 digits) | Train lengths: 1 → 5,137, 2 → 18,130, 3 → 8,691, 4 → 1,434, 5 → 9, 6 → **1**. Test tops out at 5. |
| Drop `train/29930.png` (the only 6-digit image) | One example can't teach a 6th head anything, and test has no 6-digit numbers. |
| Crop to the box around **all** digits, grown 30% | Images run from 25 to 1,083 px wide, mostly background. Cropping keeps the number big after resizing. The 30% margin is from Goodfellow et al. (2013) and keeps digit edges. |
| Clip the crop to the image | 181 train and 39 test images have a box that starts at negative pixels or runs past the edge. |
| Resize to **64×64** | A CNN needs one fixed input size. 64×64 is what the paper used. Aspect ratio is stretched, which is fine for digits. |
| SVHN label 10 → digit **0**; then **10 = blank** | SVHN stores 0 as 10. After remapping, 10 is free to mean "no digit in this position" for numbers shorter than 5. |
| Length head has classes 0–5 | `y_len` can be used directly as the class index, with no off-by-one shift. |
| 10% of train → validation (seed 42) | We tune on val and touch test once. The fixed seed gives everyone the same split. |
| Store `uint8`, scale to 0–1 in `make_dataset` | This uses 4× less memory than float32 on disk and in RAM. |

**Note:** the validation set has no 5-digit numbers (train only has 9), so val accuracy says nothing about 5-digit numbers.

**Training speed:** TensorFlow ≥ 2.11 can't use a GPU on native Windows. Use Google Colab (GPU runtime) or WSL2 for full training runs.
