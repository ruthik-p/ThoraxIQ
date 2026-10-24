# ThoraxIQ

ThoraxIQ is an educational Streamlit app that extracts HOG, LBP, and GLCM
features from an uploaded chest radiograph and scores 14 labels with the
committed multi-output Random Forest model.

> Academic use only. This project is not a medical diagnostic tool and must
> not be used to make clinical decisions.

## Streamlit Community Cloud deployment

1. Keep the app and runtime files tracked in the GitHub repository:
   - `app.py`
   - `requirements.txt`
   - `src/chestxray/`
   - `artifacts/random_forest/random_forest_model.joblib`
   - `artifacts/random_forest/random_forest_metrics.json`
2. In Streamlit Community Cloud, create an app from the
   `ruthik-p/ThoraxIQ` repository and `main` branch.
3. Set the app file path to `app.py`.
4. In Advanced settings, select Python 3.12.
5. Deploy the app.

The Random Forest model is loaded from the committed artifact at startup.
The metrics JSON is optional for inference, but is required to display the
saved held-out evaluation metrics. The feature modules under `src/chestxray/`
are required for preprocessing and feature extraction. Do not remove the model
or these source files from the repository.

The dependency versions in `requirements.txt` match the current local
`.venv` installation as a deployment compatibility candidate. They are not
confirmed records of the versions used to train or save the committed model.

## How the app works

The app accepts PNG and JPEG uploads, converts each image to 64 × 64 grayscale,
and extracts HOG, LBP, and GLCM features. It uses the committed
`MultiOutputClassifier` Random Forest artifact to produce vote scores for 14
disease labels. Saved per-label thresholds are used to show thresholded labels.

Scores are not calibrated probabilities or diagnoses. If no disease label
crosses its configured threshold, that does not establish that an image is
healthy. Uploaded images are processed for the current app session and are
not added to a training dataset.

## Model and evaluation files

The selected Random Forest model and its saved metrics are stored under
`artifacts/random_forest/`. The app checks that the model's feature version
and label order match its feature pipeline. It displays saved held-out metrics
only when the metrics file identifies the same model setup.

Other training scripts, KNN files, and experimental artifacts are not used
by this app's inference path.

## Limitations

This is an educational research demonstration. The model uses handcrafted
features and may not generalize to images outside its evaluation data. Its
performance is limited, and it must not be used to guide medical decisions.
