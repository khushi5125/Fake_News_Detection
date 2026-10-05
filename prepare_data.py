"""
prepare_data.py

Prepare the English fake-news datasets used by this project.

Inputs (inside data/):
    WELFake_Dataset.csv
    gossipcop_real.csv
    politifact_real.csv
    politifact_fake.csv

Outputs:
    data/processed/train.csv
    data/processed/val.csv
    data/processed/test.csv

Label convention used by the project:
    0 = Real
    1 = Fake

Important:
    WELFake's raw labels are the opposite:
        0 = Fake
        1 = Real
    so they are flipped here.

The split is stratified 80/10/10. Duplicates are removed BEFORE splitting
so that the same article cannot leak into train and test.
"""

from pathlib import Path
import argparse
import csv
import re
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

csv.field_size_limit(2147483647)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = DATA_DIR / "processed"
RANDOM_STATE = 42

# Which RAW WELFake label means FAKE news?
#   0 -> follows the dataset documentation (0 = fake, 1 = real)
#   1 -> use this if the automatic check printed below shows that
#        raw label 0 is actually the real (Reuters-style) news.
WELFAKE_RAW_FAKE_LABEL = 1


def clean_text(text: str) -> str:
    """Light cleaning that keeps Hindi/English characters intact."""
    if text is None or pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_welfake(path: Path) -> pd.DataFrame:
    """Read WELFake robustly even when article commas created extra columns."""
    rows = []
    skipped = 0

    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header

        for row in reader:
            # Expected: [id, title, text, label, ...]
            if len(row) < 4:
                skipped += 1
                continue

            raw_label = row[3].strip()

            # Most malformed rows still have the correct label at index 3.
            # If it was shifted, look for the first valid label in the early
            # post-text fields. This is only a recovery fallback.
            label_index = 3
            if raw_label not in {"0", "1"}:
                found = None
                for i in range(3, min(len(row), 12)):
                    if row[i].strip() in {"0", "1"}:
                        found = i
                        break
                if found is None:
                    skipped += 1
                    continue
                label_index = found

            raw_label = row[label_index].strip()

            title = row[1] if len(row) > 1 else ""
            if label_index > 2:
                # Join any comma-split pieces back into the article text.
                text = ",".join(row[2:label_index])
            else:
                text = row[2] if len(row) > 2 else ""

            # Project convention: 0 = Real, 1 = Fake.
            # WELFAKE_RAW_FAKE_LABEL says which raw value means Fake.
            label = 1 if raw_label == str(WELFAKE_RAW_FAKE_LABEL) else 0

            rows.append(
                {
                    "title": clean_text(title),
                    "text": clean_text(text),
                    "label": label,
                    "raw_label": raw_label,
                    "source": "WELFake",
                    "language": "en",
                }
            )

    df = pd.DataFrame(rows)
    print(f"WELFake: loaded {len(df):,} valid rows; skipped {skipped:,} rows")
    check_welfake_labels(df)
    return df.drop(columns=["raw_label"])


def check_welfake_labels(df: pd.DataFrame) -> None:
    """Sanity-check the WELFake label convention using Reuters wire reports.

    Articles containing '(Reuters)' are real news, so the raw label with the
    higher Reuters share should be the one mapped to Real (0).
    """
    share = {}
    print("\nWELFake label sanity check (share of articles containing '(Reuters)'):")
    for raw in ["0", "1"]:
        sub = df[df["raw_label"] == raw]
        share[raw] = sub["text"].str.contains(r"\(Reuters\)", regex=True).mean() if len(sub) else 0.0
        print(f"  raw label {raw}: n={len(sub):,}, Reuters share = {share[raw]:.1%}")

    real_raw = "1" if str(WELFAKE_RAW_FAKE_LABEL) == "0" else "0"
    other_raw = "0" if real_raw == "1" else "1"
    if share[real_raw] >= share[other_raw]:
        print(f"  OK: raw label {real_raw} is treated as REAL and has the higher Reuters share.\n")
    else:
        print("  *** WARNING: LABELS LOOK INVERTED! ***")
        print(f"  Raw label {other_raw} has the higher Reuters share (real news), but it is mapped to FAKE.")
        print(f"  Set WELFAKE_RAW_FAKE_LABEL = {1 - WELFAKE_RAW_FAKE_LABEL} at the top of this file, "
              "then re-run prepare_data.py and retrain.\n")


def load_fake_news_net(path: Path, label: int, source: str) -> pd.DataFrame:
    """Load FakeNewsNet files. These files contain reliable titles, not full article text."""
    df = pd.read_csv(path, usecols=["title"], encoding="utf-8", encoding_errors="replace")

    if "title" not in df.columns:
        raise ValueError(f"{path.name} does not contain a 'title' column")

    out = pd.DataFrame()
    out["title"] = df["title"].fillna("").astype(str).map(clean_text)
    out["text"] = ""
    out["label"] = label
    out["source"] = source
    out["language"] = "en"

    out = out[out["title"].str.len() > 5].copy()
    return out.reset_index(drop=True)


def build_input_text(df: pd.DataFrame) -> pd.DataFrame:
    """Create the single text field consumed by the Transformer."""
    df = df.copy()
    df["input_text"] = (
        df["title"].fillna("").astype(str)
        + " "
        + df["text"].fillna("").astype(str)
    ).map(clean_text)

    # Very short samples are not useful for this task.
    df = df[df["input_text"].str.len() >= 20].copy()
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact and normalized duplicate articles."""
    df = df.copy()

    # Exact duplicate text.
    df = df.drop_duplicates(subset=["input_text"])

    # Normalized duplicate text catches differences in spacing/case.
    normalized = (
        df["input_text"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    )
    df = df.loc[~normalized.duplicated()].copy()

    return df.reset_index(drop=True)


def balance(df: pd.DataFrame, max_per_class: int | None) -> pd.DataFrame:
    """Downsample the majority class; never duplicate examples."""
    counts = df["label"].value_counts()
    if not {0, 1}.issubset(counts.index):
        raise ValueError(f"Both labels 0 and 1 are required. Found: {counts.to_dict()}")

    n = min(int(counts[0]), int(counts[1]))
    if max_per_class is not None:
        n = min(n, max_per_class)

    parts = []
    for label in [0, 1]:
        part = df[df["label"] == label].sample(n=n, random_state=RANDOM_STATE)
        parts.append(part)

    result = pd.concat(parts, ignore_index=True)
    return result.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=30000,
        help="Maximum number of samples kept per class after merging (default: 30000). Use 0 for no cap.",
    )
    args = parser.parse_args()

    print("\n=== Fake News Dataset Preparation ===\n")

    welfake_path = DATA_DIR / "WELFake_Dataset.csv"
    gossip_path = DATA_DIR / "gossipcop_real.csv"
    pol_real_path = DATA_DIR / "politifact_real.csv"
    pol_fake_path = DATA_DIR / "politifact_fake.csv"

    required = [welfake_path, gossip_path, pol_real_path, pol_fake_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Missing dataset files:\n" + "\n".join(missing))

    welfake = load_welfake(welfake_path)
    gossip_real = load_fake_news_net(gossip_path, label=0, source="GossipCop")
    politifact_real = load_fake_news_net(pol_real_path, label=0, source="PolitiFact")
    politifact_fake = load_fake_news_net(pol_fake_path, label=1, source="PolitiFact")

    df = pd.concat(
        [welfake, gossip_real, politifact_real, politifact_fake],
        ignore_index=True,
    )

    df = build_input_text(df)
    before = len(df)
    df = deduplicate(df)
    print(f"Removed {before - len(df):,} duplicate rows")

    print("\nClass distribution before balancing:")
    print(df["label"].value_counts().sort_index().rename(index={0: "Real", 1: "Fake"}))

    max_per_class = None if args.max_per_class == 0 else args.max_per_class
    df = balance(df, max_per_class=max_per_class)

    print("\nClass distribution after balancing:")
    print(df["label"].value_counts().sort_index().rename(index={0: "Real", 1: "Fake"}))

    # First: 80% train, 20% temporary.
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    # Then split the 20% equally -> 10% validation, 10% test.
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["label"],
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    columns = ["input_text", "title", "text", "label", "source", "language"]

    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        part = part[columns].reset_index(drop=True)
        part.to_csv(OUT_DIR / f"{name}.csv", index=False, encoding="utf-8")
        print(f"Saved {name:5s}: {len(part):,} rows -> {OUT_DIR / (name + '.csv')}")

    print("\nDone. Run translate_augment.py next if Hindi augmentation is desired.")


if __name__ == "__main__":
    main()