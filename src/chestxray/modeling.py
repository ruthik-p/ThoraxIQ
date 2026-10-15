"""KNN training, validation-based threshold selection, and evaluation."""
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_pipeline(k: int, pca_components: float | None = None, weights: str = "distance",
                  metric: str = "euclidean") -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if pca_components is not None:
        steps.append(("pca", PCA(n_components=pca_components, svd_solver="full")))
    # KNeighborsClassifier supports multi-output targets directly. A single
    # index avoids storing the same training matrix once for every disease.
    steps.append(("knn", KNeighborsClassifier(n_neighbors=k, weights=weights,
                                               metric=metric, n_jobs=-1)))
    return Pipeline(steps)


def probabilities(model: Pipeline, x: np.ndarray) -> np.ndarray:
    output = []
    classifier = model.named_steps["knn"]
    transformed = model[:-1].transform(x)
    for classes, proba in zip(classifier.classes_, classifier.predict_proba(transformed)):
        positive_class = np.flatnonzero(classes == 1)
        output.append(proba[:, positive_class[0]] if len(positive_class) else np.zeros(len(proba)))
    return np.column_stack(output)


def tune_thresholds(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Choose each label's F1-maximizing threshold on validation data."""
    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)
    candidates = np.arange(0.10, 0.91, 0.05)
    for col in range(y_true.shape[1]):
        if y_true[:, col].sum() == 0:
            continue
        values = [f1_score(y_true[:, col], scores[:, col] >= threshold, zero_division=0)
                  for threshold in candidates]
        thresholds[col] = candidates[int(np.argmax(values))]
    return thresholds


def evaluate(y_true: np.ndarray, scores: np.ndarray, thresholds: np.ndarray, label_names: list[str]):
    predicted = scores >= thresholds
    report = classification_report(y_true, predicted, target_names=label_names, zero_division=0,
                                   output_dict=True)
    per_label = precision_recall_fscore_support(y_true, predicted, average=None, zero_division=0)
    aucs = {}
    for i, label in enumerate(label_names):
        try:
            aucs[label] = float(roc_auc_score(y_true[:, i], scores[:, i]))
        except ValueError:
            aucs[label] = None
    return {
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, predicted, average="micro", zero_division=0)),
        "classification_report": report,
        "confusion_matrices": {name: confusion_matrix(y_true[:, i], predicted[:, i], labels=[0, 1]).tolist()
                               for i, name in enumerate(label_names)},
        "per_label": {name: {"precision": float(per_label[0][i]), "recall": float(per_label[1][i]),
                              "f1": float(per_label[2][i]), "support": int(per_label[3][i]),
                              "positive_count": int(y_true[:, i].sum()),
                              "roc_auc": aucs[name]} for i, name in enumerate(label_names)},
    }


def save_model(path: str | Path, model, labels, thresholds, feature_version: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": list(labels), "thresholds": np.asarray(thresholds),
                 "feature_version": feature_version}, path)

