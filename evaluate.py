"""
evaluate.py

Evaluates the fine-tuned XLM-RoBERTa model on the held-out test set.

Labels:
    0 = Real
    1 = Fake

Run:
    python evaluate.py
"""

from pathlib import Path

import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from transformers import (
    AutoConfig,
    pipeline,
)


BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "fake_news_model"
TEST_PATH = BASE_DIR / "data" / "processed" / "test.csv"


def convert_prediction_to_id(prediction, config):
    """
    Convert the model's predicted label into:
        0 = Real
        1 = Fake
    """

    label = str(prediction["label"]).strip().upper()

    # Check model configuration first
    id2label = {
        int(k): str(v).upper()
        for k, v in config.id2label.items()
    }

    # Match the returned label against model configuration
    for idx, config_label in id2label.items():

        if label == config_label:
            return idx

    # Common Hugging Face labels
    if label in ["LABEL_0", "REAL", "REAL NEWS", "0"]:
        return 0

    if label in ["LABEL_1", "FAKE", "FAKE NEWS", "1"]:
        return 1

    raise ValueError(
        f"Unknown model label: {prediction['label']}\n"
        f"Model id2label: {id2label}"
    )


def main():

    # ---------------------------------------------------------
    # 1. Check model
    # ---------------------------------------------------------

    if not MODEL_DIR.exists():
        raise SystemExit(
            "fake_news_model/ not found. "
            "Run prepare_data.py then train.py first."
        )

    # ---------------------------------------------------------
    # 2. Check test dataset
    # ---------------------------------------------------------

    if not TEST_PATH.exists():
        raise SystemExit(
            "data/processed/test.csv not found. "
            "Run prepare_data.py first."
        )

    # ---------------------------------------------------------
    # 3. Load test data
    # ---------------------------------------------------------

    df = pd.read_csv(TEST_PATH)

    print(
        f"\nEvaluating on {len(df)} held-out test rows "
        "(never seen during training)"
    )

    print("\nTest-set label distribution:")

    print(
        df["label"]
        .value_counts(normalize=True)
        .rename("share")
    )

    # ---------------------------------------------------------
    # 4. Load model configuration
    # ---------------------------------------------------------

    print("\nLoading model configuration...")

    config = AutoConfig.from_pretrained(
        str(MODEL_DIR)
    )

    print("Model id2label:")

    print(config.id2label)

    # ---------------------------------------------------------
    # 5. Select device
    # ---------------------------------------------------------

    if torch.cuda.is_available():

        device = 0

        print(
            f"\nUsing GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        device = -1

        print("\nUsing CPU")

    # ---------------------------------------------------------
    # 6. Load fine-tuned model
    # ---------------------------------------------------------

    print("\nLoading fine-tuned model...")

    classifier = pipeline(
        "text-classification",
        model=str(MODEL_DIR),
        tokenizer=str(MODEL_DIR),
        device=device,
        truncation=True,
        max_length=256,
    )

    print("Model loaded successfully.")

    # ---------------------------------------------------------
    # 7. Run predictions
    # ---------------------------------------------------------

    print(
        f"\nRunning predictions on "
        f"{len(df)} test articles..."
    )

    predictions = classifier(
        df["input_text"].tolist(),
        batch_size=64,
        truncation=True,
    )

    # ---------------------------------------------------------
    # 8. Convert predictions to 0/1
    # ---------------------------------------------------------

    predicted_labels = [
        convert_prediction_to_id(
            prediction,
            config
        )
        for prediction in predictions
    ]

    true_labels = df["label"].astype(int).tolist()

    # ---------------------------------------------------------
    # 9. Calculate metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    precision = precision_score(
        true_labels,
        predicted_labels,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        pos_label=1,
        zero_division=0
    )

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1]
    )

    # ---------------------------------------------------------
    # 10. Display results
    # ---------------------------------------------------------

    print("\n")
    print("=" * 55)
    print("          HELD-OUT TEST RESULTS")
    print("=" * 55)

    print(
        f"\nAccuracy  : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision : {precision * 100:.2f}% (Fake)"
    )

    print(
        f"Recall    : {recall * 100:.2f}% (Fake)"
    )

    print(
        f"F1 Score  : {f1 * 100:.2f}% (Fake)"
    )

    # ---------------------------------------------------------
    # 11. Confusion matrix
    # ---------------------------------------------------------

    print("\nConfusion Matrix [[TN, FP], [FN, TP]]:")

    print(cm)

    # ---------------------------------------------------------
    # 12. Classification report
    # ---------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=[0, 1],
            target_names=[
                "Real (0)",
                "Fake (1)"
            ],
            zero_division=0,
        )
    )

    # ---------------------------------------------------------
    # 13. Accuracy by source
    # ---------------------------------------------------------

    if "source" in df.columns:

        print("\n")
        print("=" * 55)
        print("                 BY SOURCE")
        print("=" * 55)

        prediction_series = pd.Series(
            predicted_labels,
            index=df.index
        )

        for src in df["source"].unique():

            mask = df["source"] == src

            sub_true = df.loc[
                mask,
                "label"
            ]

            sub_pred = prediction_series.loc[
                mask
            ]

            acc = accuracy_score(
                sub_true,
                sub_pred
            )

            print(
                f"{src:16} "
                f"(n={mask.sum():5d}): "
                f"Accuracy = {acc * 100:.2f}%"
            )

    # ---------------------------------------------------------
    # 14. Save results
    # ---------------------------------------------------------

    results_file = MODEL_DIR / "test_results.txt"

    with open(
        results_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "HELD-OUT TEST RESULTS\n"
        )

        f.write(
            "=====================\n\n"
        )

        f.write(
            f"Test samples: {len(df)}\n"
        )

        f.write(
            f"Accuracy: {accuracy * 100:.2f}%\n"
        )

        f.write(
            f"Precision (Fake): "
            f"{precision * 100:.2f}%\n"
        )

        f.write(
            f"Recall (Fake): "
            f"{recall * 100:.2f}%\n"
        )

        f.write(
            f"F1 Score (Fake): "
            f"{f1 * 100:.2f}%\n\n"
        )

        f.write(
            "Confusion Matrix:\n"
        )

        f.write(
            str(cm)
        )

        f.write(
            "\n\nClassification Report:\n"
        )

        f.write(
            classification_report(
                true_labels,
                predicted_labels,
                labels=[0, 1],
                target_names=[
                    "Real (0)",
                    "Fake (1)"
                ],
                zero_division=0,
            )
        )

    print(
        f"\nResults saved to:\n"
        f"{results_file}"
    )

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()