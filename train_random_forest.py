"""Validation-selected multi-label Random Forest trainer for ThoraxIQ.

Reads the existing feature cache and patient split, then writes RF-specific
artifacts. The held-out test split is evaluated only after validation selection.
"""
import hashlib
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier

from src.chestxray.data import encode_labels, load_metadata, patient_splits
from src.chestxray.features import FEATURE_VERSION
from src.chestxray.modeling import evaluate, tune_thresholds


def load_verified_cache(frame: pd.DataFrame, cache_path: Path, image_dir: Path):
    """Load cached features only when its source signature and row map match."""
    names = frame["Image Index"].astype(str).tolist()
    signature = [str(image_dir.resolve())]
    for name in names:
        try:
            stat = (image_dir / name).stat()
            signature.append(f"{name}:{stat.st_size}:{stat.st_mtime_ns}")
        except OSError:
            signature.append(f"{name}:missing")
    key = hashlib.sha256(
        "\n".join(names).encode() + FEATURE_VERSION.encode()
        + "\n".join(signature).encode()
    ).hexdigest()
    with np.load(cache_path, allow_pickle=False) as cache:
        features = cache["features"]
        valid = cache["valid_indices"].astype(int)
        errors = cache["errors"].astype(str)
        cached_key = str(cache["key"].item())
    if cached_key != key:
        raise ValueError("Feature cache is stale for the current CSV/images; no cache rebuild was attempted.")
    if len(errors) or not np.array_equal(valid, np.arange(len(frame))):
        raise ValueError("Feature cache has missing/unreadable image rows; refusing misaligned training data.")
    if features.shape != (len(frame), 1790) or not np.isfinite(features).all():
        raise ValueError(f"Unexpected feature cache shape or non-finite values: {features.shape}")
    return features


def rf_scores(model, features):
    """Return positive-class scores in the fixed 14-label output order."""
    per_label = model.predict_proba(features)
    columns = []
    for classes, values in zip(model.classes_, per_label):
        positive = np.flatnonzero(classes == 1)
        columns.append(values[:, positive[0]] if len(positive) else np.zeros(len(values)))
    return np.column_stack(columns)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="labels.csv")
    parser.add_argument("--images", default="images")
    parser.add_argument("--cache", default="artifacts/features.npz")
    parser.add_argument("--output", default="artifacts/random_forest_retrained")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output)
    protected = {
        Path("artifacts/random_forest").resolve(),
        Path("artifacts/random_forest_experiments/rf_tuning_20261015").resolve(),
    }
    if output.resolve() in protected:
        raise ValueError("Choose a new output directory; existing RF artifacts are protected.")
    output.mkdir(parents=True, exist_ok=True)
    frame = load_metadata(args.csv)
    labels, y = encode_labels(frame)
    features = load_verified_cache(frame, Path(args.cache), Path(args.images))
    train_idx, val_idx, test_idx = patient_splits(frame, args.seed)

    candidates = [
        {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 100, "max_depth": 24, "min_samples_leaf": 2},
        {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 3},
    ]
    selected = None
    search = []
    warning_records = []
    total_fit_seconds = 0.0
    for params in candidates:
        base = RandomForestClassifier(
            **params, max_features="sqrt", class_weight="balanced_subsample",
            random_state=args.seed, n_jobs=1,
        )
        model = MultiOutputClassifier(base, n_jobs=-1)
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(features[train_idx], y[train_idx])
        fit_seconds = time.perf_counter() - start
        total_fit_seconds += fit_seconds
        warning_records.extend(str(item.message) for item in caught)
        val_start = time.perf_counter()
        val_scores = rf_scores(model, features[val_idx])
        val_prediction_seconds = time.perf_counter() - val_start
        val_thresholds = tune_thresholds(y[val_idx], val_scores)
        val_macro_f1 = float(f1_score(y[val_idx], val_scores >= val_thresholds,
                                      average="macro", zero_division=0))
        record = {**params, "validation_macro_f1": val_macro_f1,
                  "fit_seconds": fit_seconds,
                  "validation_predict_seconds": val_prediction_seconds}
        search.append(record)
        print(f"Candidate {params}: validation macro F1={val_macro_f1:.4f}; fit={fit_seconds:.1f}s")
        if selected is None or val_macro_f1 > selected[0]:
            selected = (val_macro_f1, params, model, val_thresholds)

    validation_macro_f1, params, model, thresholds = selected
    # Selection is complete. This is the only point at which the test split is read.
    test_start = time.perf_counter()
    test_scores = rf_scores(model, features[test_idx])
    test_prediction_seconds = time.perf_counter() - test_start
    results = evaluate(y[test_idx], test_scores, thresholds, labels)

    model_path = output / "random_forest_model.joblib"
    joblib.dump({"model": model, "labels": labels, "thresholds": thresholds,
                 "feature_version": FEATURE_VERSION, "model_type": "RandomForest"}, model_path)
    model_size = model_path.stat().st_size

    # Reload and verify label/score dimensions and numerical agreement.
    loaded = joblib.load(model_path)
    if loaded["labels"] != labels or len(loaded["thresholds"]) != len(labels):
        raise RuntimeError("Reloaded RF artifact label/threshold alignment check failed.")
    reload_start = time.perf_counter()
    reloaded_scores = rf_scores(loaded["model"], features[test_idx[:min(32, len(test_idx))]])
    reload_prediction_seconds = time.perf_counter() - reload_start
    if reloaded_scores.shape != (min(32, len(test_idx)), len(labels)) or not np.allclose(
            reloaded_scores, test_scores[:len(reloaded_scores)], rtol=0, atol=0):
        raise RuntimeError("Reloaded RF model scores do not match saved-model scores.")

    results.update({
        "model_type": "RandomForestClassifier per label via MultiOutputClassifier",
        "selected_parameters": params,
        "validation_macro_f1": validation_macro_f1,
        "thresholds": thresholds.tolist(),
        "split_seed": args.seed,
        "feature_version": FEATURE_VERSION,
        "split_rows": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "split_patients": {name: int(frame.iloc[idx]["Patient ID"].nunique())
                           for name, idx in (("train", train_idx), ("validation", val_idx),
                                             ("test", test_idx))},
        "validation_search": search,
        "timing_seconds": {"training_all_candidates": total_fit_seconds,
                           "test_prediction": test_prediction_seconds,
                           "reload_prediction_first_32": reload_prediction_seconds},
        "model_size_bytes": model_size,
        "warnings": sorted(set(warning_records)),
        "labels": labels,
    })
    (output / "random_forest_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame(results["per_label"]).T.to_csv(output / "random_forest_per_label_metrics.csv")
    pd.DataFrame(results["classification_report"]).T.to_csv(
        output / "random_forest_classification_report.csv")

    print(f"Selected {params}; validation macro F1={validation_macro_f1:.4f}")
    print(f"RF test macro F1={results['macro_f1']:.4f}; micro F1={results['micro_f1']:.4f}")
    print(f"Saved RF model/report only under {output}; existing models were not changed.")


if __name__ == "__main__":
    main()
