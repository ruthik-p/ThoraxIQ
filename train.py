"""Train and evaluate a patient-separated handcrafted-feature KNN model."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.chestxray.data import encode_labels, load_metadata, patient_splits
from src.chestxray.features import FEATURE_VERSION, extract_dataset
from src.chestxray.modeling import (evaluate, make_pipeline, probabilities, save_model,
                                    tune_thresholds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="labels.csv")
    parser.add_argument("--images", default="images", help="Image directory or images.zip")
    parser.add_argument("--cache", default="artifacts/features.npz")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca", type=float, default=None, help="Optional PCA fraction, e.g. 0.95")
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    frame = load_metadata(args.csv)
    duplicate_summary = {
        "exact_duplicate_rows_removed": frame.attrs.get("exact_duplicate_rows_removed", 0),
        "duplicate_image_rows_removed": frame.attrs.get("duplicate_image_rows_removed", 0),
    }
    label_names, y = encode_labels(frame)
    print(f"Rows: {len(frame)} | patients: {frame['Patient ID'].nunique()} | labels: {label_names}")
    print("Finding distribution:\n", frame["Finding Labels"].value_counts().head(20).to_string())
    features, valid, errors = extract_dataset(frame, args.images, args.cache, args.force_features)
    if len(errors):
        (output / "image_errors.txt").write_text("\n".join(errors), encoding="utf-8")
        print(f"Skipped {len(errors)} missing/corrupt images; see {output / 'image_errors.txt'}")
    frame, y = frame.iloc[valid].reset_index(drop=True), y[valid]
    train_idx, val_idx, test_idx = patient_splits(frame, args.seed)
    print(f"Split rows: train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}")

    best = None
    validation_search = []
    for k in (3, 5, 7, 9):
        for weights in ("distance", "uniform"):
            for distance_metric in ("euclidean", "manhattan"):
                model = make_pipeline(k, args.pca, weights, distance_metric)
                model.fit(features[train_idx], y[train_idx])
                score = probabilities(model, features[val_idx])
                thresholds = tune_thresholds(y[val_idx], score)
                predicted = score >= thresholds
                metric = float(__import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(
                    y[val_idx], predicted, average="macro", zero_division=0))
                validation_search.append({"k": k, "weights": weights,
                                          "distance_metric": distance_metric,
                                          "validation_macro_f1": metric})
                print(f"k={k}, weights={weights}, metric={distance_metric}: "
                      f"validation macro F1={metric:.4f}")
                if best is None or metric > best[0]:
                    best = (metric, k, weights, distance_metric, model, thresholds)

    _, best_k, best_weights, best_metric, model, thresholds = best
    test_scores = probabilities(model, features[test_idx])
    results = evaluate(y[test_idx], test_scores, thresholds, label_names)
    qualitative_rows = []
    seen_truths = set()
    for test_position, original_idx in enumerate(test_idx):
        truth = frame.iloc[original_idx]["Finding Labels"]
        if truth in {"No Finding", *label_names} and truth not in seen_truths:
            scores = test_scores[test_position]
            predicted = [label for label, score, threshold in
                         zip(label_names, scores, thresholds) if score >= threshold]
            record = {"image": frame.iloc[original_idx]["Image Index"],
                      "ground_truth": truth, "predicted_labels": "|".join(predicted)}
            record.update({f"score_{label}": float(score)
                           for label, score in zip(label_names, scores)})
            qualitative_rows.append(record)
            seen_truths.add(truth)
        if len(qualitative_rows) >= 5:
            break
    pd.DataFrame(qualitative_rows).to_csv(output / "test_sample_predictions.csv", index=False)
    results.update({"selected_k": best_k, "selected_weights": best_weights,
                    "selected_distance_metric": best_metric,
                    "validation_macro_f1": best[0], "thresholds": thresholds.tolist(),
                    "split_seed": args.seed, "feature_version": FEATURE_VERSION,
                    "test_rows": int(len(test_idx)), "duplicate_summary": duplicate_summary,
                    "validation_search": validation_search,
                    "split_rows": {"train": int(len(train_idx)), "validation": int(len(val_idx)),
                                   "test": int(len(test_idx))},
                    "split_patients": {name: int(frame.iloc[idx]["Patient ID"].nunique())
                                       for name, idx in (("train", train_idx), ("validation", val_idx),
                                                         ("test", test_idx))},
                    "label_counts_by_split": {
                        name: {label: int(y[idx, col].sum()) for col, label in enumerate(label_names)}
                        for name, idx in (("train", train_idx), ("validation", val_idx),
                                          ("test", test_idx))}})

    # Re-evaluate the preserved baseline on the exact same patient-separated
    # test rows and features used for the new model.
    backups = sorted(output.glob("baseline_backup_*/knn_model.joblib"), reverse=True)
    baseline_path = Path(backups[0]) if backups else None
    comparison = {"baseline_model": str(baseline_path) if baseline_path else None,
                  "test_split_seed": args.seed, "same_test_rows": int(len(test_idx)),
                  "new_model": {"macro_f1": results["macro_f1"], "micro_f1": results["micro_f1"],
                                "selected_k": best_k, "weights": best_weights,
                                "distance_metric": best_metric}}
    if baseline_path is not None:
        import joblib
        baseline = joblib.load(baseline_path)
        if baseline.get("feature_version") != FEATURE_VERSION:
            comparison["baseline_note"] = "Feature version differs; baseline comparison skipped."
        else:
            baseline_scores = probabilities(baseline["model"], features[test_idx])
            baseline_results = evaluate(y[test_idx], baseline_scores,
                                        np.asarray(baseline["thresholds"]), label_names)
            comparison["baseline"] = {"macro_f1": baseline_results["macro_f1"],
                                      "micro_f1": baseline_results["micro_f1"]}
            per_label_comparison = {}
            for label in label_names:
                old = baseline_results["per_label"][label]
                new = results["per_label"][label]
                per_label_comparison[label] = {
                    "baseline_precision": old["precision"], "new_precision": new["precision"],
                    "baseline_recall": old["recall"], "new_recall": new["recall"],
                    "baseline_f1": old["f1"], "new_f1": new["f1"],
                    "f1_delta_new_minus_baseline": new["f1"] - old["f1"],
                    "baseline_roc_auc": old["roc_auc"], "new_roc_auc": new["roc_auc"],
                    "test_positive_count": new["positive_count"],
                }
            comparison["per_label"] = per_label_comparison
            pd.DataFrame(per_label_comparison).T.to_csv(output / "baseline_comparison.csv")
            comparison["macro_f1_delta_new_minus_baseline"] = (
                results["macro_f1"] - baseline_results["macro_f1"])
            comparison["micro_f1_delta_new_minus_baseline"] = (
                results["micro_f1"] - baseline_results["micro_f1"])
    (output / "baseline_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame(results["per_label"]).T.to_csv(output / "per_label_metrics.csv")
    pd.DataFrame(results["classification_report"]).T.to_csv(output / "classification_report.csv")
    labels = list(results["per_label"])
    plt.figure(figsize=(12, 5))
    plt.bar(labels, [results["per_label"][name]["f1"] for name in labels])
    plt.xticks(rotation=55, ha="right")
    plt.ylabel("F1 score")
    plt.title("Held-out test F1 by disease")
    plt.tight_layout()
    plt.savefig(output / "disease_f1.png", dpi=160)
    plt.close()
    save_model(output / "knn_model.joblib", model, label_names, thresholds, FEATURE_VERSION)
    print(f"Test macro F1={results['macro_f1']:.4f}; micro F1={results['micro_f1']:.4f}")
    print(f"Saved model and reports in {output}")


if __name__ == "__main__":
    main()

