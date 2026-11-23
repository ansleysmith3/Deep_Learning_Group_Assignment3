import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

from data_pipeline.config import MAX_DIGITS
from data_pipeline.load_data import HEAD_NAMES, decode, make_dataset

# ==========================================
# 1. Hyperparameters & Knobs (For Person 3)
# ==========================================
CONFIG = {
    "batch_size": 64,
    "learning_rate": 1e-3,
    "epochs": 15,
    "conv_filters": [32, 64, 128],
    "dense_units": 256,
    "dropout_rate": 0.3,
    "fast_debug": False,  # Set to False for full training on ISAAC GPU
}

os.makedirs("artifacts", exist_ok=True)


# ==========================================
# 2. Build Multi-Output CNN (Functional API)
# ==========================================
def build_svhn_model(input_shape=(64, 64, 3), config=CONFIG):
    inputs = layers.Input(shape=input_shape, name="image_input")
    x = inputs

    # Shared Convolutional Feature Extractor
    for idx, filters in enumerate(config["conv_filters"], start=1):
        x = layers.Conv2D(
            filters, kernel_size=3, padding="same", activation="relu", name=f"conv{idx}_1"
        )(x)
        x = layers.Conv2D(
            filters, kernel_size=3, padding="same", activation="relu", name=f"conv{idx}_2"
        )(x)
        x = layers.MaxPooling2D(pool_size=2, name=f"pool{idx}")(x)
        x = layers.BatchNormalization(name=f"bn{idx}")(x)

    # Shared Dense Representation
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(config["dense_units"], activation="relu", name="shared_dense")(x)
    x = layers.Dropout(config["dropout_rate"], name="shared_dropout")(x)

    # k + 1 Softmax Output Heads (must match HEAD_NAMES from load_data.py)
    outputs = {
        "length": layers.Dense(MAX_DIGITS + 1, activation="softmax", name="length")(x)
    }
    for i in range(1, MAX_DIGITS + 1):
        head_name = f"digit{i}"
        outputs[head_name] = layers.Dense(11, activation="softmax", name=head_name)(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="SVHN_MultiDigit_CNN")
    return model


# ==========================================
# 3. Compile & Train Baseline
# ==========================================
def train_baseline(config=CONFIG):
    train_ds = make_dataset("train", batch_size=config["batch_size"], shuffle=True)
    val_ds = make_dataset("val", batch_size=config["batch_size"], shuffle=False)

    if config["fast_debug"]:
        print(">> FAST_DEBUG is ON: Using 20 train batches & 5 val batches for CPU verification.")
        train_ds = train_ds.take(20)
        val_ds = val_ds.take(5)
        epochs = 3
    else:
        epochs = config["epochs"]

    model = build_svhn_model(config=config)
    model.summary()

    # One sparse categorical cross-entropy loss & accuracy metric per head
    losses = {name: "sparse_categorical_crossentropy" for name in HEAD_NAMES}
    metrics = {name: "accuracy" for name in HEAD_NAMES}

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss=losses,
        metrics=metrics,
    )

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath="artifacts/baseline_best.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint_cb],
        verbose=1,
    )
    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed:.1f}s ({elapsed / epochs:.1f}s per epoch).")

    # Save training curves and history for slides & Person 3
    save_training_curves(history.history)
    with open("artifacts/baseline_history.json", "w") as f:
        json.dump(history.history, f, indent=2)

    return model, history


# ==========================================
# 4. Save Training Curves for Slides
# ==========================================
def save_training_curves(hist):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Total Loss
    axes[0].plot(hist["loss"], label="Train Total Loss")
    axes[0].plot(hist["val_loss"], label="Val Total Loss")
    axes[0].set_title("Total Loss (Sum of 6 Heads)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Sparse Categorical Cross-Entropy")
    axes[0].legend()

    # Per-Head Validation Accuracy
    for name in HEAD_NAMES:
        axes[1].plot(hist[f"val_{name}_accuracy"], label=f"val_{name}")
    axes[1].set_title("Validation Accuracy by Head")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("artifacts/baseline_curves.png", dpi=150)
    print("Saved training curves to artifacts/baseline_curves.png")


# ==========================================
# 5. Prediction Step (Decoding Full Numbers)
# ==========================================
def predict_house_numbers(model, dataset, num_batches=-1):
    """
    Runs inference on batches from dataset (num_batches=-1 evaluates all batches),
    extracts argmax per head, and uses load_data.decode() to reconstruct full house number strings.
    """
    results = []
    ds = dataset if num_batches is None else dataset.take(num_batches)
    for x_batch, y_true_batch in ds:
        preds = model.predict(x_batch, verbose=0)  # dict of (batch, num_classes) arrays

        pred_len = np.argmax(preds["length"], axis=1)
        pred_digits = np.column_stack(
            [np.argmax(preds[f"digit{i}"], axis=1) for i in range(1, MAX_DIGITS + 1)]
        )

        true_len = y_true_batch["length"].numpy()
        true_digits = np.column_stack(
            [y_true_batch[f"digit{i}"].numpy() for i in range(1, MAX_DIGITS + 1)]
        )

        for i in range(len(pred_len)):
            pred_str = decode(pred_len[i], pred_digits[i])
            true_str = decode(true_len[i], true_digits[i])
            results.append((true_str, pred_str))

    return results


if __name__ == "__main__":
    model, history = train_baseline(CONFIG)

    val_ds = make_dataset("val", batch_size=64, shuffle=False)
    # num_batches=None evaluates the entire dataset (all 3,340 val images)
    all_preds = predict_house_numbers(model, val_ds, num_batches=-1)

    correct = sum(1 for true_str, pred_str in all_preds if true_str == pred_str)
    whole_acc = correct / len(all_preds)
    print(f"\nFull Validation Whole-Number Accuracy: {whole_acc:.4f} ({correct}/{len(all_preds)})")

    print("\nSample Validation Predictions (True -> Predicted):")
    for true_str, pred_str in all_preds[:10]:
        print(f"  True: {true_str:<6} | Pred: {pred_str:<6}")