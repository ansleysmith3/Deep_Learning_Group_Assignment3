import tensorflow as tf
from data_pipeline.load_data import make_dataset
from model_baseline_ximin_v2 import predict_house_numbers

# Load the best checkpoint saved during your 42-minute run
best_model = tf.keras.models.load_model("artifacts/baseline_best.keras")

val_ds = make_dataset("val", batch_size=64, shuffle=False)
all_preds = predict_house_numbers(best_model, val_ds, num_batches=-1)

correct = sum(1 for true_str, pred_str in all_preds if true_str == pred_str)
whole_acc = correct / len(all_preds)
print(f"\nFull Validation Whole-Number Accuracy: {whole_acc:.4f} ({correct}/{len(all_preds)})")

print("\nSample Validation Predictions (True -> Predicted):")
for true_str, pred_str in all_preds[:15]:
    print(f"  True: {true_str:<6} | Pred: {pred_str:<6}")