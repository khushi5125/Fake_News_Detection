"""
translate_augment.py

Optional stopgap Hindi augmentation.
Translates a sample of the English training set to Hindi.
"""

from pathlib import Path
import argparse
import random
import time

import pandas as pd

try:
    from deep_translator import GoogleTranslator
except ImportError:
    raise SystemExit(
        "deep-translator is not installed. Run: pip install deep-translator"
    )

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "data" / "processed" / "train.csv"
RANDOM_STATE = 42


def translate_one(translator, text: str, retries: int = 4) -> str:
    """Translate one article with safe slicing and exponential backoff."""
    text = str(text).strip()
    # Shorten payload to prevent edge timeouts and character-limit throttling
    text_for_translation = text[:1500]

    for attempt in range(retries):
        try:
            translated = translator.translate(text_for_translation)
            if translated and translated.strip():
                return translated.strip()
        except Exception as exc:
            # Wait longer on 429/Too Many Requests blocks
            wait_time = (attempt + 1) * 4  # 4s, 8s, 12s, 16s
            if "Too many requests" in str(exc) or "Server Error" in str(exc):
                wait_time += 6

            if attempt == retries - 1:
                print(f"  translation failed: {exc}")
            else:
                time.sleep(wait_time)

    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.10,
        help="Fraction of the training rows to translate (default: 0.10).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=500,
        help="Maximum number of translated rows (default: 500).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.6,
        help="Delay between successful requests (default: 0.6 sec to stay < 2 req/sec).",
    )
    args = parser.parse_args()

    if not 0 < args.fraction <= 1:
        raise SystemExit("--fraction must be > 0 and <= 1")

    if not TRAIN_PATH.exists():
        raise SystemExit(
            "data/processed/train.csv not found. Run prepare_data.py first."
        )

    df = pd.read_csv(TRAIN_PATH, encoding="utf-8")

    required = {"input_text", "label", "source", "language"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"train.csv is missing columns: {sorted(missing)}")

    candidates = df[df["language"].fillna("en") == "en"].copy()

    requested = min(
        args.max_rows,
        max(1, int(len(candidates) * args.fraction)),
    )

    samples = []
    per_class = requested // 2

    for label in [0, 1]:
        class_df = candidates[candidates["label"] == label]
        n = min(per_class, len(class_df))
        if n:
            samples.append(
                class_df.sample(n=n, random_state=RANDOM_STATE + label)
            )

    selected = pd.concat(samples, ignore_index=True) if samples else candidates.head(0)

    remaining_n = requested - len(selected)
    if remaining_n > 0:
        remaining = candidates.drop(index=selected.index, errors="ignore")
        if len(remaining):
            selected = pd.concat(
                [
                    selected,
                    remaining.sample(
                        n=min(remaining_n, len(remaining)),
                        random_state=RANDOM_STATE,
                    ),
                ],
                ignore_index=True,
            )

    selected = selected.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"Training rows available: {len(df):,}")
    print(f"Hindi rows requested:    {len(selected):,}")
    print("\nStarting machine translation. This requires internet access.\n")

    translator = GoogleTranslator(source="en", target="hi")

    translated_rows = []
    for i, (_, row) in enumerate(selected.iterrows(), start=1):
        translated = translate_one(translator, row["input_text"])

        if translated:
            new_row = row.copy()
            new_row["input_text"] = translated
            new_row["title"] = translated
            new_row["text"] = ""
            new_row["language"] = "hi"
            new_row["source"] = str(row["source"]) + "_machine_translated_hi"
            translated_rows.append(new_row)

        if i % 10 == 0 or i == len(selected):
            print(f"Processed {i}/{len(selected)} | successful: {len(translated_rows)}")

        time.sleep(max(0.2, args.sleep))

    if not translated_rows:
        raise SystemExit(
            "No translations were produced. Check your internet connection or translator service."
        )

    translated_df = pd.DataFrame(translated_rows)

    existing = set(df["input_text"].astype(str).str.strip())
    translated_df = translated_df[
        ~translated_df["input_text"].astype(str).str.strip().isin(existing)
    ].copy()
    translated_df = translated_df.drop_duplicates(subset=["input_text"])

    combined = pd.concat([df, translated_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    combined.to_csv(TRAIN_PATH, index=False, encoding="utf-8")

    print(f"\nAdded {len(translated_df):,} Hindi training rows.")
    print(f"Updated training file: {TRAIN_PATH}")


if __name__ == "__main__":
    main()