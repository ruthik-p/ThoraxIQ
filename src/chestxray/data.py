"""CSV loading, label encoding, and patient-level dataset splits."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

LABEL_COLUMN = "Finding Labels"
IMAGE_COLUMN = "Image Index"
PATIENT_COLUMN = "Patient ID"
DEFAULT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]


def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate the NIH-style metadata CSV."""
    frame = pd.read_csv(csv_path)
    required = {IMAGE_COLUMN, LABEL_COLUMN, PATIENT_COLUMN}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    missing_rows = frame[[IMAGE_COLUMN, LABEL_COLUMN, PATIENT_COLUMN]].isna().any(axis=1)
    if missing_rows.any():
        raise ValueError(f"CSV has {int(missing_rows.sum())} row(s) missing an image name, finding label, or patient ID")
    frame = frame.copy()
    frame[IMAGE_COLUMN] = frame[IMAGE_COLUMN].astype(str)
    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(str)
    # Exact duplicate CSV rows add weight to those images in KNN. Remove them,
    # and reject conflicting labels for one filename rather than guessing.
    input_rows = len(frame)
    frame = frame.drop_duplicates().reset_index(drop=True)
    exact_duplicates_removed = input_rows - len(frame)
    grouped = frame.groupby(IMAGE_COLUMN).agg(
        label_values=(LABEL_COLUMN, "nunique"), patient_values=(PATIENT_COLUMN, "nunique"))
    conflicts = grouped[(grouped["label_values"] > 1) | (grouped["patient_values"] > 1)]
    if not conflicts.empty:
        raise ValueError("Conflicting labels or patient IDs for image filename(s): " +
                         ", ".join(conflicts.index[:10].astype(str)))
    duplicate_images_removed = frame[IMAGE_COLUMN].duplicated().sum()
    frame = frame.drop_duplicates(subset=[IMAGE_COLUMN], keep="first").reset_index(drop=True)
    frame.attrs["exact_duplicate_rows_removed"] = int(exact_duplicates_removed)
    frame.attrs["duplicate_image_rows_removed"] = int(duplicate_images_removed)
    return frame


def encode_labels(frame: pd.DataFrame, label_names: list[str] | None = None):
    """Encode pipe-separated findings in a stable label order; No Finding is all zero."""
    labels = list(label_names or DEFAULT_LABELS)
    y = np.zeros((len(frame), len(labels)), dtype=np.uint8)
    known = set(labels) | {"No Finding"}
    for row, value in enumerate(frame[LABEL_COLUMN]):
        findings = {item.strip() for item in value.split("|") if item.strip()}
        if not findings:
            raise ValueError(f"Empty finding label at CSV row {row + 2}")
        if "No Finding" in findings and len(findings) > 1:
            raise ValueError(f"No Finding is combined with disease labels at CSV row {row + 2}")
        unknown = findings - known
        if unknown:
            raise ValueError(f"Unknown finding label(s): {sorted(unknown)}")
        for col, label in enumerate(labels):
            y[row, col] = int(label in findings)
    return labels, y


def patient_splits(frame: pd.DataFrame, seed: int = 42):
    """Return row indices for approximate 70/15/15 splits, grouped by patient."""
    groups = frame[PATIENT_COLUMN].astype(str).to_numpy()
    first = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, remainder_idx = next(first.split(np.zeros(len(frame)), groups=groups))
    remainder_groups = groups[remainder_idx]
    second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    val_rel, test_rel = next(second.split(np.zeros(len(remainder_idx)), groups=remainder_groups))
    val_idx, test_idx = remainder_idx[val_rel], remainder_idx[test_rel]
    sets = [set(groups[idx]) for idx in (train_idx, val_idx, test_idx)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise RuntimeError("Patient leakage detected between data splits")
    return train_idx, val_idx, test_idx

