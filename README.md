# Feature-Based Multi-Label Chest X-ray Classification Using KNN

An academic machine-learning project using HOG, LBP, and GLCM image features with a multi-output K-Nearest Neighbors classifier. It does not use CNNs, SVMs, or deep-learning models.

> Academic use only. This project is not a medical diagnostic tool and must not be used to make clinical decisions.

## Data layout

The workspace contains `labels.csv` and 14,999 extracted PNG images under `images/`. Training reads that folder directly; it does not require the original ZIP. The CSV must contain `Image Index`, `Finding Labels`, and `Patient ID` columns. Exact duplicate rows are removed, duplicate filenames with conflicting labels or patient IDs are rejected, and missing required values are reported as an error.

## Setup

Use Python 3.10 or newer. In PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Python 3.12 and the required packages were installed for this project in `.venv`. For a fresh setup, install Python 3.10 or newer, create the environment, and install the dependencies with the commands above.

## Train

```powershell
.\.venv\Scripts\python.exe train.py --csv labels.csv --images images --seed 42
```

Feature extraction caches in `artifacts/features.npz`. Use `--force-features` to rebuild it. Reports and model artifacts are written to `artifacts/`. Optional PCA can be enabled with `--pca 0.95`; it is fitted inside the training pipeline. Run the initial implementation on the full set only if your machine has sufficient time and memory; feature extraction and KNN can take substantial time.

The trainer validates the CSV, records unreadable images, makes approximate 70/15/15 patient-group splits, and compares `k` values 3, 5, 7, and 9; uniform and distance weights; and Euclidean and Manhattan distances using validation macro F1. The final test set is not used to select the model. StandardScaler (and optional PCA) is fitted on training rows only. Per-label thresholds are selected on validation data. A validation-only PCA trial retaining 95% variance used 479 components and scored macro F1 `0.0974`, below the selected no-PCA score `0.1094`, so the final model retains all 1,790 handcrafted features. The feature cache includes each source image's size and modification time in its key. Matplotlib uses a non-GUI backend so report generation works on headless machines.

## Web application

After training:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The interface accepts PNG, JPG, and JPEG files, displays the grayscale image at the model's 64x64 input size, and shows the score, validation threshold, and decision for all 14 diseases. It clearly explains when no labels cross their thresholds. Scores are KNN neighbor-based estimates, not calibrated disease probabilities.

## Optional Random Forest experiment

`train_random_forest.py` reuses the verified `artifacts/features.npz` cache and the same patient split (seed 42). It selects among independent per-label Random Forest configurations using validation data and writes its model and reports under `artifacts/random_forest/`. It does not overwrite the KNN model. The website continues to use KNN by default; after explicitly choosing to try the saved Random Forest, set `THORAXIQ_MODEL=random_forest` in the environment before starting Streamlit. Unset the variable or set it to `knn` to use the existing KNN model.

## Method

- **HOG:** captures local edge and shape structure.
- **LBP:** summarizes local texture patterns.
- **GLCM:** computes contrast, homogeneity, energy, and correlation across four directions.
- **KNN:** classifies each label output using distance-weighted nearest neighbors.
- **No Finding:** encoded as 14 zeros. The disease outputs are not a separate mutually exclusive class.

The feature cache stores only successful image rows plus their original row indices. Images are processed individually. The model artifact includes the label order, fitted pipeline, thresholds, and feature version.

## Outputs and limitations

Training on the extracted data processed all 14,999 images; none were missing or unreadable. The CSV has 14 disease labels and 3,923 patients. On the patient-separated split with seed 42, the selected configuration was `k=9`, distance weighting, and Manhattan distance, with validation macro F1 `0.1094`. On the held-out 2,376-row test set, macro F1 was `0.0915` and micro F1 was `0.1501`; ROC-AUC was calculable for all labels. The prior model was evaluated on the same test rows using the same feature vectors: macro F1 `0.0967`, micro F1 `0.1414`. The new model's macro F1 is lower by `0.0053`; micro F1 is higher by `0.0088`. Per-disease comparison is saved in `artifacts/baseline_comparison.csv`. These results are weak and not suitable for clinical use.

Training creates `knn_model.joblib`, `metrics.json`, `per_label_metrics.csv`, `classification_report.csv`, `disease_f1.png`, `baseline_comparison.json`/`.csv`, and `test_sample_predictions.csv`. The metrics include test positive counts and each label's 2x2 confusion matrix. ROC-AUC is null when the held-out test set does not contain both classes for that label. The previous model and reports were preserved in `artifacts/baseline_backup_20261014_204624/` before retraining. The dataset is imbalanced (for example, 46 Hernia annotations versus 2,260 Infiltration annotations) and labels can be noisy. Handcrafted features discard much of the original image detail, and KNN inference may be slow as the training set grows. Patient-level splitting prevents the same patient's rows crossing data splits, but does not remove other sources of dataset bias.

The saved model reloaded successfully and produced 14 scores for multiple held-out real images with different CSV labels. Streamlit's upload widget was exercised with `sample_images/00000324_013.png`; the app returned all 14 disease rows and the expected no-label-crossed-threshold message, without script errors. A malformed PNG upload was also handled with an in-app error message. That sample image is annotated `No Finding`; a no-positive prediction is not proof that an image is healthy. The model is an educational experiment, not a medical diagnostic system.

The earlier app displayed only labels that crossed thresholds, which made a zero-label result look like missing output. The underlying scores are weighted neighbor votes and can be zero or low when nearby training images do not share a label. The app now displays every disease score and its validation threshold. That makes output easier to inspect but does not make the model clinically reliable.

