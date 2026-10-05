# ============================================================
# FAKE NEWS DETECTION - XLM-RoBERTa TRAINING
# ============================================================
#
# Features:
#   - XLM-RoBERTa-base
#   - Automatic Google Drive checkpoint saving
#   - Automatic resume after Colab disconnect/restart
#   - Class-weighted CrossEntropyLoss
#   - Label smoothing
#   - Early stopping
#   - Mixed precision (FP16/BF16)
#   - Accuracy / Balanced Accuracy
#   - Fake Precision / Recall / F1
#   - Confusion Matrix
#   - Classification Report
#   - Best model restoration
#
# Expected project structure:
#
# Fake_news_detection/
# ├── train.py
# ├── data/
# │   └── processed/
# │       ├── train.csv
# │       ├── val.csv
# │       └── test.csv
# └── fake_news_model/
#     └── checkpoints/
#
# Labels:
#   0 = Real
#   1 = Fake
#
# ============================================================


from pathlib import Path
import argparse
import random
import shutil

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from transformers.trainer_utils import get_last_checkpoint


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"

MODEL_DIR = BASE_DIR / "fake_news_model"

CHECKPOINT_DIR = MODEL_DIR / "checkpoints"


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "xlm-roberta-base"

NUM_LABELS = 2

LABEL_NAMES = {
    0: "Real",
    1: "Fake",
}


# ============================================================
# 3. TRAINING DEFAULTS
# ============================================================

DEFAULT_LEARNING_RATE = 2e-5

DEFAULT_TRAIN_BATCH_SIZE = 4

DEFAULT_EVAL_BATCH_SIZE = 8

DEFAULT_GRADIENT_ACCUMULATION = 4

DEFAULT_EPOCHS = 5

DEFAULT_MAX_LENGTH = 256

DEFAULT_PATIENCE = 2

RANDOM_STATE = 42


# ============================================================
# 4. CHECKPOINT CONFIGURATION
# ============================================================

# Save checkpoints more frequently because Colab sessions
# can disconnect unexpectedly.

CHECKPOINT_SAVE_STEPS = 250

CHECKPOINT_EVAL_STEPS = 250

CHECKPOINT_LOGGING_STEPS = 100

# Keep only the latest 3 checkpoints.
CHECKPOINT_SAVE_LIMIT = 3


# ============================================================
# 5. RANDOM SEED
# ============================================================

def set_seed(seed=RANDOM_STATE):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 6. DATASET CLASS
# ============================================================

class FakeNewsDataset(torch.utils.data.Dataset):

    def __init__(self, dataframe, tokenizer, max_length):

        self.texts = dataframe["input_text"].tolist()

        self.labels = dataframe["label"].astype(int).tolist()

        self.tokenizer = tokenizer

        self.max_length = max_length

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, idx):

        text = str(self.texts[idx])

        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoding.items()
        }

        item["labels"] = torch.tensor(
            label,
            dtype=torch.long
        )

        return item


# ============================================================
# 7. LOAD DATA
# ============================================================

def load_csv(path):

    print(f"\nLoading dataset:")
    print(path)

    if not path.exists():

        raise FileNotFoundError(
            f"\nDataset file not found:\n{path}\n"
        )

    df = pd.read_csv(path)

    print(f"Rows loaded: {len(df):,}")

    required_columns = {
        "input_text",
        "label"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:

        raise ValueError(
            f"\nMissing required columns in {path}:\n"
            f"{missing_columns}\n\n"
            f"Required columns:\n"
            f"input_text\n"
            f"label"
        )

    # Remove missing text
    df = df.dropna(
        subset=["input_text"]
    ).copy()

    # Convert text to string
    df["input_text"] = (
        df["input_text"]
        .astype(str)
        .str.strip()
    )

    # Remove empty text
    df = df[
        df["input_text"].str.len() > 0
    ].copy()

    # Convert labels
    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    # Remove invalid labels
    df = df.dropna(
        subset=["label"]
    ).copy()

    df["label"] = df["label"].astype(int)

    # Keep only binary labels
    df = df[
        df["label"].isin([0, 1])
    ].copy()

    df = df.reset_index(drop=True)

    print(f"Clean rows: {len(df):,}")

    print(
        "Label distribution:"
    )

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    return df


# ============================================================
# 8. METRICS
# ============================================================

def make_metrics(eval_prediction):

    predictions = eval_prediction.predictions

    labels = eval_prediction.label_ids

    if isinstance(predictions, tuple):

        predictions = predictions[0]

    predictions = np.argmax(
        predictions,
        axis=-1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        labels,
        predictions
    )

    fake_precision = precision_score(
        labels,
        predictions,
        pos_label=1,
        zero_division=0
    )

    fake_recall = recall_score(
        labels,
        predictions,
        pos_label=1,
        zero_division=0
    )

    fake_f1 = f1_score(
        labels,
        predictions,
        pos_label=1,
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "fake_precision": fake_precision,
        "fake_recall": fake_recall,
        "fake_f1": fake_f1,
    }


# ============================================================
# 9. WEIGHTED TRAINER
# ============================================================

class WeightedTrainer(Trainer):

    def __init__(
        self,
        *args,
        class_weights=None,
        label_smoothing=0.0,
        **kwargs
    ):

        super().__init__(
            *args,
            **kwargs
        )

        self.class_weights = class_weights

        self.label_smoothing = label_smoothing

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):

        labels = inputs.pop("labels")

        outputs = model(**inputs)

        logits = outputs.logits

        if self.class_weights is not None:

            weights = self.class_weights.to(
                logits.device
            )

        else:

            weights = None

        loss_function = torch.nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=self.label_smoothing,
        )

        loss = loss_function(
            logits,
            labels
        )

        return (
            loss,
            outputs
        ) if return_outputs else loss


# ============================================================
# 10. CALCULATE CLASS WEIGHTS
# ============================================================

def calculate_class_weights(train_df):

    counts = (
        train_df["label"]
        .value_counts()
        .sort_index()
    )

    count_real = counts.get(0, 0)

    count_fake = counts.get(1, 0)

    total = count_real + count_fake

    if count_real == 0 or count_fake == 0:

        raise ValueError(
            "Training dataset must contain both "
            "Real (0) and Fake (1) labels."
        )

    weight_real = total / (
        2.0 * count_real
    )

    weight_fake = total / (
        2.0 * count_fake
    )

    weights = torch.tensor(
        [
            weight_real,
            weight_fake
        ],
        dtype=torch.float
    )

    print("\nClass weights:")

    print(
        f"Real (0): {weight_real:.4f}"
    )

    print(
        f"Fake (1): {weight_fake:.4f}"
    )

    return weights


# ============================================================
# 11. FIND LATEST CHECKPOINT
# ============================================================

def find_latest_checkpoint():

    if not CHECKPOINT_DIR.exists():

        return None

    try:

        checkpoint = get_last_checkpoint(
            str(CHECKPOINT_DIR)
        )

        return checkpoint

    except Exception as e:

        print(
            f"\nWarning: Could not detect checkpoint: {e}"
        )

        return None


# ============================================================
# 12. SHOW CHECKPOINT STATUS
# ============================================================

def show_checkpoint_status():

    print("\n" + "=" * 60)

    print("CHECKPOINT STATUS")

    print("=" * 60)

    print(
        f"Checkpoint directory:\n"
        f"{CHECKPOINT_DIR}"
    )

    if not CHECKPOINT_DIR.exists():

        print(
            "\nNo checkpoint directory exists yet."
        )

        return None

    checkpoints = []

    for path in CHECKPOINT_DIR.glob(
        "checkpoint-*"
    ):

        if path.is_dir():

            try:

                step = int(
                    path.name.split("-")[-1]
                )

                checkpoints.append(
                    (step, path)
                )

            except ValueError:

                continue

    if not checkpoints:

        print(
            "\nNo checkpoints found."
        )

        return None

    checkpoints.sort(
        key=lambda x: x[0]
    )

    print(
        f"\nFound {len(checkpoints)} checkpoint(s):"
    )

    for step, path in checkpoints:

        print(
            f"  checkpoint-{step}"
        )

    latest = checkpoints[-1][1]

    print(
        f"\nLatest checkpoint:"
    )

    print(latest)

    return str(latest)


# ============================================================
# 13. COMMAND LINE ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train XLM-RoBERTa for "
            "Fake News Detection"
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="Hugging Face model name"
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum token length"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate"
    )

    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=DEFAULT_TRAIN_BATCH_SIZE,
        help="Training batch size per device"
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=DEFAULT_EVAL_BATCH_SIZE,
        help="Evaluation batch size per device"
    )

    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION,
        help="Gradient accumulation steps"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="Early stopping patience"
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Ignore existing checkpoints "
            "and start training from scratch"
        )
    )

    return parser.parse_args()


# ============================================================
# 14. MAIN TRAINING FUNCTION
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(RANDOM_STATE)

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("\n")

    print("=" * 70)

    print(
        "FAKE NEWS DETECTION - "
        "XLM-RoBERTa TRAINING"
    )

    print("=" * 70)

    print(
        f"\nProject directory:\n{BASE_DIR}"
    )

    print(
        f"\nProcessed data directory:\n"
        f"{PROCESSED_DIR}"
    )

    print(
        f"\nModel directory:\n"
        f"{MODEL_DIR}"
    )

    print(
        f"\nCheckpoint directory:\n"
        f"{CHECKPOINT_DIR}"
    )

    print(
        f"\nModel:\n{args.model}"
    )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("DEVICE CONFIGURATION")

    print("=" * 70)

    use_cuda = torch.cuda.is_available()

    print(
        f"\nCUDA available: {use_cuda}"
    )

    if use_cuda:

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        gpu_memory = (
            torch.cuda.get_device_properties(0)
            .total_memory
            / (1024 ** 3)
        )

        print(
            f"GPU memory: "
            f"{gpu_memory:.2f} GB"
        )

    else:

        print(
            "\nWARNING: CUDA is not available."
        )

        print(
            "Training will run on CPU."
        )

    # --------------------------------------------------------
    # Mixed precision
    # --------------------------------------------------------

    use_bf16 = (
        use_cuda
        and torch.cuda.is_bf16_supported()
    )

    use_fp16 = (
        use_cuda
        and not use_bf16
    )

    print(
        f"\nBF16: {use_bf16}"
    )

    print(
        f"FP16: {use_fp16}"
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("LOADING DATASETS")

    print("=" * 70)

    train_path = (
        PROCESSED_DIR / "train.csv"
    )

    val_path = (
        PROCESSED_DIR / "val.csv"
    )

    test_path = (
        PROCESSED_DIR / "test.csv"
    )

    train_df = load_csv(
        train_path
    )

    val_df = load_csv(
        val_path
    )

    test_df = load_csv(
        test_path
    )

    print("\nDataset sizes:")

    print(
        f"Train: {len(train_df):,}"
    )

    print(
        f"Validation: {len(val_df):,}"
    )

    print(
        f"Test: {len(test_df):,}"
    )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weights = (
        calculate_class_weights(
            train_df
        )
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("LOADING TOKENIZER")

    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model
    )

    print(
        "\nTokenizer loaded successfully."
    )

    # --------------------------------------------------------
    # Dataset objects
    # --------------------------------------------------------

    print("\nCreating tokenized datasets...")

    train_ds = FakeNewsDataset(
        train_df,
        tokenizer,
        args.max_length
    )

    val_ds = FakeNewsDataset(
        val_df,
        tokenizer,
        args.max_length
    )

    test_ds = FakeNewsDataset(
        test_df,
        tokenizer,
        args.max_length
    )

    print(
        "Datasets created successfully."
    )

    # --------------------------------------------------------
    # Model configuration
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("LOADING MODEL")

    print("=" * 70)

    config = AutoConfig.from_pretrained(
        args.model,
        num_labels=NUM_LABELS,
        id2label={
            0: "REAL",
            1: "FAKE"
        },
        label2id={
            "REAL": 0,
            "FAKE": 1
        },
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        config=config
    )

    print(
        "\nXLM-RoBERTa model loaded successfully."
    )

    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("CREATING TRAINING CONFIGURATION")

    print("=" * 70)

    training_args = TrainingArguments(

        output_dir=str(
            CHECKPOINT_DIR
        ),

        # Evaluation
        eval_strategy="steps",

        eval_steps=CHECKPOINT_EVAL_STEPS,

        # Checkpoint saving
        save_strategy="steps",

        save_steps=CHECKPOINT_SAVE_STEPS,

        save_total_limit=CHECKPOINT_SAVE_LIMIT,

        # Logging
        logging_strategy="steps",

        logging_steps=CHECKPOINT_LOGGING_STEPS,

        # Optimization
        learning_rate=args.learning_rate,

        weight_decay=0.01,

        warmup_steps=500,

        num_train_epochs=args.epochs,

        # Batch sizes
        per_device_train_batch_size=(
            args.train_batch_size
        ),

        per_device_eval_batch_size=(
            args.eval_batch_size
        ),

        # Gradient accumulation
        gradient_accumulation_steps=(
            args.gradient_accumulation
        ),

        # Best model
        load_best_model_at_end=True,

        metric_for_best_model=(
            "eval_fake_f1"
        ),

        greater_is_better=True,

        # Mixed precision
        fp16=use_fp16,

        bf16=use_bf16,

        # Data loader
        dataloader_pin_memory=use_cuda,

        # Reproducibility
        seed=RANDOM_STATE,

        data_seed=RANDOM_STATE,

        # Disable external logging
        report_to="none",

        # IMPORTANT:
        # save_safetensors=True has been removed
        # because your current Transformers version
        # does not support it in TrainingArguments.
    )

    print(
        "\nTraining configuration created."
    )

    print(
        f"\nCheckpoint interval: "
        f"{CHECKPOINT_SAVE_STEPS} steps"
    )

    print(
        f"Maximum saved checkpoints: "
        f"{CHECKPOINT_SAVE_LIMIT}"
    )

    # --------------------------------------------------------
    # Create trainer
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("CREATING TRAINER")

    print("=" * 70)

    trainer = WeightedTrainer(

        model=model,

        args=training_args,

        train_dataset=train_ds,

        eval_dataset=val_ds,

        processing_class=tokenizer,

        compute_metrics=make_metrics,

        class_weights=class_weights,

        label_smoothing=0.05,

        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.patience
            )
        ],
    )

    print(
        "\nTrainer created successfully."
    )

    # --------------------------------------------------------
    # Check checkpoint status
    # --------------------------------------------------------

    latest_checkpoint = (
        show_checkpoint_status()
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("STARTING TRAINING")

    print("=" * 70)

    if args.fresh:

        print(
            "\n--fresh was specified."
        )

        print(
            "Existing checkpoints will be ignored."
        )

        print(
            "\nStarting training from scratch..."
        )

        train_result = trainer.train()

    elif latest_checkpoint:

        print(
            "\nRESUMING FROM CHECKPOINT"
        )

        print(
            f"\nCheckpoint:"
        )

        print(
            latest_checkpoint
        )

        print(
            "\nTraining will continue from "
            "the latest saved checkpoint."
        )

        train_result = trainer.train(
            resume_from_checkpoint=(
                latest_checkpoint
            )
        )

    else:

        print(
            "\nNo checkpoint found."
        )

        print(
            "Starting training from scratch..."
        )

        train_result = trainer.train()

    # --------------------------------------------------------
    # Training complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("TRAINING COMPLETED")

    print("=" * 70)

    print(
        f"\nTraining loss:"
    )

    print(
        train_result.training_loss
    )

    # --------------------------------------------------------
    # Save training metrics
    # --------------------------------------------------------

    train_metrics = (
        train_result.metrics
    )

    trainer.log_metrics(
        "train",
        train_metrics
    )

    trainer.save_metrics(
        "train",
        train_metrics
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("VALIDATION EVALUATION")

    print("=" * 70)

    eval_metrics = trainer.evaluate(
        eval_dataset=val_ds
    )

    print(
        "\nValidation results:"
    )

    for key, value in eval_metrics.items():

        if isinstance(value, float):

            print(
                f"{key}: {value:.4f}"
            )

        else:

            print(
                f"{key}: {value}"
            )

    trainer.log_metrics(
        "eval",
        eval_metrics
    )

    trainer.save_metrics(
        "eval",
        eval_metrics
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("FINAL TEST EVALUATION")

    print("=" * 70)

    test_output = trainer.predict(
        test_ds
    )

    test_predictions = (
        np.argmax(
            test_output.predictions,
            axis=-1
        )
    )

    test_labels = (
        test_output.label_ids
    )

    # --------------------------------------------------------
    # Test metrics
    # --------------------------------------------------------

    test_accuracy = accuracy_score(
        test_labels,
        test_predictions
    )

    test_balanced_accuracy = (
        balanced_accuracy_score(
            test_labels,
            test_predictions
        )
    )

    test_precision = precision_score(
        test_labels,
        test_predictions,
        pos_label=1,
        zero_division=0
    )

    test_recall = recall_score(
        test_labels,
        test_predictions,
        pos_label=1,
        zero_division=0
    )

    test_f1 = f1_score(
        test_labels,
        test_predictions,
        pos_label=1,
        zero_division=0
    )

    # --------------------------------------------------------
    # Print test results
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("FINAL TEST RESULTS")

    print("=" * 70)

    print(
        f"\nAccuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy: "
        f"{test_balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Fake Precision: "
        f"{test_precision * 100:.2f}%"
    )

    print(
        f"Fake Recall: "
        f"{test_recall * 100:.2f}%"
    )

    print(
        f"Fake F1 Score: "
        f"{test_f1 * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        test_labels,
        test_predictions
    )

    print("\n" + "=" * 70)

    print("CONFUSION MATRIX")

    print("=" * 70)

    print(
        "\n                 Predicted"
    )

    print(
        "                 Real    Fake"
    )

    print(
        f"Actual Real     "
        f"{cm[0][0]:6d}  "
        f"{cm[0][1]:6d}"
    )

    print(
        f"Actual Fake     "
        f"{cm[1][0]:6d}  "
        f"{cm[1][1]:6d}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("CLASSIFICATION REPORT")

    print("=" * 70)

    report = classification_report(
        test_labels,
        test_predictions,
        target_names=[
            "Real",
            "Fake"
        ],
        digits=4,
        zero_division=0
    )

    print(
        "\n"
        + report
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("SAVING FINAL MODEL")

    print("=" * 70)

    trainer.save_model(
        str(MODEL_DIR)
    )

    tokenizer.save_pretrained(
        str(MODEL_DIR)
    )

    print(
        f"\nFinal model saved to:"
    )

    print(
        MODEL_DIR
    )

    # --------------------------------------------------------
    # Save test results
    # --------------------------------------------------------

    results_file = (
        MODEL_DIR / "test_results.txt"
    )

    with open(
        results_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "FAKE NEWS DETECTION\n"
        )

        f.write(
            "XLM-RoBERTa-base\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            f"Model: {args.model}\n"
        )

        f.write(
            f"Max Length: "
            f"{args.max_length}\n"
        )

        f.write(
            f"Learning Rate: "
            f"{args.learning_rate}\n"
        )

        f.write(
            f"Epochs: "
            f"{args.epochs}\n"
        )

        f.write(
            f"Train Batch Size: "
            f"{args.train_batch_size}\n"
        )

        f.write(
            f"Evaluation Batch Size: "
            f"{args.eval_batch_size}\n"
        )

        f.write(
            f"Gradient Accumulation: "
            f"{args.gradient_accumulation}\n"
        )

        f.write(
            "\n"
        )

        f.write(
            "FINAL TEST METRICS\n"
        )

        f.write(
            "-" * 60 + "\n"
        )

        f.write(
            f"Accuracy: "
            f"{test_accuracy * 100:.2f}%\n"
        )

        f.write(
            f"Balanced Accuracy: "
            f"{test_balanced_accuracy * 100:.2f}%\n"
        )

        f.write(
            f"Fake Precision: "
            f"{test_precision * 100:.2f}%\n"
        )

        f.write(
            f"Fake Recall: "
            f"{test_recall * 100:.2f}%\n"
        )

        f.write(
            f"Fake F1 Score: "
            f"{test_f1 * 100:.2f}%\n"
        )

        f.write(
            "\n"
        )

        f.write(
            "CONFUSION MATRIX\n"
        )

        f.write(
            "-" * 60 + "\n"
        )

        f.write(
            f"{cm}\n"
        )

        f.write(
            "\n"
        )

        f.write(
            "CLASSIFICATION REPORT\n"
        )

        f.write(
            "-" * 60 + "\n"
        )

        f.write(
            report
        )

    print(
        f"\nTest results saved to:"
    )

    print(
        results_file
    )

    # --------------------------------------------------------
    # Final checkpoint status
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("FINAL CHECKPOINT STATUS")

    print("=" * 70)

    show_checkpoint_status()

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("ALL DONE")

    print("=" * 70)

    print(
        "\nYour trained model is located at:"
    )

    print(
        MODEL_DIR
    )

    print(
        "\nYour checkpoints are located at:"
    )

    print(
        CHECKPOINT_DIR
    )

    print(
        "\nIf Google Colab disconnects:"
    )

    print(
        "1. Remount Google Drive."
    )

    print(
        "2. Go to the project directory."
    )

    print(
        "3. Run: python train.py"
    )

    print(
        "4. The latest checkpoint will "
        "automatically be detected."
    )

    print(
        "\nTo intentionally start a completely "
        "new training run:"
    )

    print(
        "python train.py --fresh"
    )

    print("\n")


# ============================================================
# 15. RUN MAIN
# ============================================================

if __name__ == "__main__":

    main()