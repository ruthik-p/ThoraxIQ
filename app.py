"""ThoraxIQ educational dashboard for the original Random Forest artifact."""
from __future__ import annotations

from hashlib import sha256
from html import escape
import io
import json
import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(APP_DIR / "artifacts" / "mplconfig"))

import joblib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
from sklearn.multioutput import MultiOutputClassifier

from src.chestxray.data import DEFAULT_LABELS
from src.chestxray.features import FEATURE_VERSION, extract_features, preprocess_image


MODEL_PATH = APP_DIR / "artifacts" / "random_forest" / "random_forest_model.joblib"
METRICS_PATH = APP_DIR / "artifacts" / "random_forest" / "random_forest_metrics.json"
SUPPORTED_TYPES = ["png", "jpg", "jpeg"]

st.set_page_config(
    page_title="ThoraxIQ | Radiograph Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
      --night: #07111f; --panel: #0d1b2b; --panel-hi: #102238;
      --line: rgba(142, 174, 198, .17); --muted: #91a6ba;
      --ink: #eaf4fa; --teal: #39d7c8; --cyan: #58c9ff;
    }
    .stApp {
      background:
        radial-gradient(ellipse at 82% 0%, rgba(19, 108, 131, .20), transparent 32%),
        radial-gradient(ellipse at 0% 28%, rgba(25, 75, 119, .16), transparent 30%),
        linear-gradient(155deg, #07111f 0%, #091726 56%, #07121f 100%);
      color: var(--ink);
    }
    [data-testid="stHeader"] { background: rgba(7, 17, 31, .55); }
    [data-testid="stToolbar"] { right: 1rem; }
    .block-container { max-width: 1370px; padding: 2rem 2rem 3rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: -.025em; }
    p, li, label, .stCaption { color: #c3d2df; }
    .hero {
      display: flex; align-items: center; justify-content: space-between; gap: 1.5rem;
      padding: 1.25rem 1.45rem; margin: .15rem 0 1.2rem;
      border: 1px solid rgba(101, 205, 220, .22); border-radius: 22px;
      background: linear-gradient(118deg, rgba(16, 38, 58, .96), rgba(10, 26, 42, .82));
      box-shadow: 0 18px 50px rgba(0, 0, 0, .22), inset 0 1px rgba(255,255,255,.035);
    }
    .brand { display: flex; align-items: center; gap: 1rem; min-width: 0; }
    .brand-mark {
      width: 54px; height: 54px; flex: 0 0 54px; display: grid; place-items: center;
      border-radius: 17px; background: linear-gradient(145deg, rgba(57,215,200,.17), rgba(88,201,255,.08));
      border: 1px solid rgba(79, 221, 213, .34); box-shadow: 0 0 30px rgba(57,215,200,.11);
    }
    .brand-title { font-size: 1.55rem; line-height: 1.15; font-weight: 750; color: #f0f8fc; }
    .brand-subtitle { margin-top: .28rem; color: #9fb6c8; font-size: .94rem; }
    .hero-right { display: flex; flex-direction: column; align-items: flex-end; gap: .55rem; }
    .status-pill {
      display: inline-flex; align-items: center; gap: .5rem; border-radius: 999px;
      padding: .48rem .8rem; background: rgba(37, 192, 158, .11);
      border: 1px solid rgba(57, 215, 200, .31); color: #8ff0d5; font-size: .82rem; font-weight: 650;
      white-space: nowrap;
    }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #48e0b6; box-shadow: 0 0 12px #48e0b6; }
    .disclaimer-top { font-size: .73rem; color: #8da3b6; text-align: right; }
    .eyebrow { text-transform: uppercase; letter-spacing: .15em; color: #64d6d0; font-size: .72rem; font-weight: 700; }
    .section-title { color: #edf6fb; font-size: 1.17rem; font-weight: 700; margin: .2rem 0 .2rem; }
    .section-copy { color: #9db1c2; font-size: .88rem; margin-bottom: .9rem; }
    .metric-card {
      height: 100%; min-height: 118px; padding: 1rem 1.08rem; border-radius: 17px;
      border: 1px solid var(--line); background: linear-gradient(145deg, rgba(16,34,52,.94), rgba(11,25,40,.91));
      box-shadow: 0 12px 32px rgba(0,0,0,.14); position: relative; overflow: hidden;
    }
    .metric-card:after { content:""; position:absolute; top:-36px; right:-24px; width:90px; height:90px;
      background: radial-gradient(circle, rgba(57,215,200,.12), transparent 70%); }
    .metric-label { color:#91aabd; font-size:.75rem; text-transform:uppercase; letter-spacing:.10em; }
    .metric-value { color:#f0f8fc; font-size:1.56rem; font-weight:740; margin-top:.48rem; line-height:1.15; }
    .metric-note { color:#8fa6b9; font-size:.76rem; margin-top:.35rem; }
    .soft-panel {
      border: 1px solid var(--line); border-radius: 19px;
      background: linear-gradient(145deg, rgba(14,31,49,.92), rgba(10,24,39,.92));
      padding: 1.15rem 1.25rem; box-shadow: 0 16px 38px rgba(0,0,0,.14);
    }
    .upload-hint { border-radius: 14px; padding: .85rem .95rem; background: rgba(26,55,75,.5);
      border:1px solid rgba(104,159,188,.18); color:#a9bdcc; font-size:.82rem; line-height:1.5; }
    .file-meta { color:#98adbf; font-size:.82rem; padding:.45rem 0 .2rem; overflow-wrap:anywhere; }
    [data-testid="stFileUploader"] section { border: 1px dashed rgba(80,202,211,.43); border-radius: 15px;
      background: rgba(12,32,49,.48); }
    [data-testid="stFileUploader"] section:hover { border-color: rgba(88,201,255,.8); }
    .stButton > button[kind="primary"] {
      border: 1px solid rgba(84,235,218,.42); border-radius: 12px; color:#04151c;
      background: linear-gradient(105deg, #3bd7c6, #66d7f5); font-weight:750;
      min-height: 2.8rem; box-shadow: 0 8px 24px rgba(49,202,190,.15); transition: transform .16s, box-shadow .16s;
    }
    .stButton > button[kind="primary"]:hover { transform: translateY(-1px); box-shadow: 0 11px 29px rgba(49,202,190,.26); }
    .stButton > button:not([kind="primary"]) { border-radius:11px; border:1px solid var(--line); background:rgba(20,39,57,.7); color:#c6d5e0; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 13px; overflow: hidden; }
    [data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 15px; background:rgba(12,27,43,.65); }
    hr { border-color: rgba(142,174,198,.15); }
    .footer {
      margin-top: 2.2rem; padding: 1.15rem 0 .2rem; border-top:1px solid var(--line);
      display:flex; justify-content:space-between; gap:1rem; color:#8299ac; font-size:.78rem;
    }
    @media (max-width: 760px) {
      .block-container { padding: 1rem .8rem 2rem; }
      .hero { align-items:flex-start; flex-direction:column; padding:1rem; }
      .hero-right { align-items:flex-start; }
      .disclaimer-top { text-align:left; }
      .brand-title { font-size:1.35rem; }
      .footer { flex-direction:column; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_selected_model(model_path: str, modified_ns: int) -> dict:
    """Load only the selected original RF artifact; cache by file revision."""
    del modified_ns  # Included in Streamlit's cache key to invalidate changed files.
    bundle = joblib.load(model_path)
    if bundle.get("feature_version") != FEATURE_VERSION:
        raise ValueError("The selected model's feature definition does not match this app.")
    if bundle.get("labels") != DEFAULT_LABELS:
        raise ValueError("The selected model's label order does not match the 14 supported diseases.")
    if len(bundle.get("thresholds", [])) != len(DEFAULT_LABELS):
        raise ValueError("The selected model does not contain 14 prediction thresholds.")
    if not isinstance(bundle.get("model"), MultiOutputClassifier):
        raise TypeError("The configured artifact is not the expected multi-output Random Forest.")
    if not hasattr(bundle["model"], "predict_proba"):
        raise TypeError("The selected Random Forest does not provide prediction scores.")
    return bundle


def positive_class_scores(model, features: np.ndarray) -> np.ndarray:
    """Get each Random Forest output's score for class 1 in label order."""
    outputs = model.predict_proba(features)
    columns = []
    for classes, values in zip(model.classes_, outputs):
        positive = np.flatnonzero(classes == 1)
        columns.append(values[:, positive[0]] if len(positive) else np.zeros(len(values)))
    scores = np.column_stack(columns)
    if scores.shape != (len(features), len(DEFAULT_LABELS)):
        raise ValueError(f"Expected 14 output scores, received {scores.shape}.")
    return scores


def metric_card(label: str, value: str, note: str = "") -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{escape(label)}</div>'
        f'<div class="metric-value">{escape(value)}</div>'
        f'<div class="metric-note">{escape(note)}</div>'
        '</div>'
    )


def check_metrics_association(metrics: dict, bundle: dict, model_bytes: int) -> tuple[bool, str]:
    """Accept reports only when metadata identifies this exact saved RF setup."""
    if metrics.get("model_type") != "RandomForestClassifier per label via MultiOutputClassifier":
        return False, "The report does not identify the expected multi-output Random Forest."
    if metrics.get("feature_version") != bundle.get("feature_version"):
        return False, "The report's feature version differs from the loaded model."
    if metrics.get("labels") != bundle.get("labels"):
        return False, "The report's disease order differs from the loaded model."
    if metrics.get("model_size_bytes") != model_bytes:
        return False, "The report's saved-model size does not match the loaded artifact."
    estimator = bundle["model"].estimator
    for key, value in metrics.get("selected_parameters", {}).items():
        if estimator.get_params().get(key) != value:
            return False, f"The report's {key} parameter does not match the loaded model."
    if len(metrics.get("per_label", {})) != len(DEFAULT_LABELS):
        return False, "The report does not contain metrics for all 14 diseases."
    return True, ""


def make_score_chart(labels: list[str], scores: np.ndarray, thresholds: np.ndarray):
    order = np.argsort(scores)
    names = [labels[i].replace("_", " ") for i in order]
    values = scores[order]
    cuts = thresholds[order]
    above = values >= cuts
    colors = ["#45d9cb" if flag else "#71879c" for flag in above]
    fig, ax = plt.subplots(figsize=(9.8, 6.5), constrained_layout=True)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ypos = np.arange(len(names))
    ax.barh(ypos, np.ones(len(names)), color="#142a3d", height=.62, zorder=1)
    ax.barh(ypos, values, color=colors, height=.43, zorder=2)
    for y, threshold in zip(ypos, cuts):
        ax.plot(threshold, y, marker="|", markersize=16, markeredgewidth=2.2,
                color="#67cfff", zorder=4)
    ax.set_yticks(ypos, names, color="#d5e2eb", fontsize=9.5)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 5), ["0", ".25", ".50", ".75", "1.0"])
    ax.tick_params(axis="x", colors="#9db2c2", labelsize=9, length=0, pad=8)
    ax.tick_params(axis="y", length=0, pad=9)
    ax.set_xlabel("Random Forest prediction score", color="#9db2c2", labelpad=12)
    ax.grid(axis="x", color="#607b91", alpha=.23, linewidth=.7)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_axisbelow(True)
    legend = [
        Patch(facecolor="#45d9cb", label="At / above threshold"),
        Patch(facecolor="#71879c", label="Below threshold"),
        Line2D([0], [0], color="#67cfff", marker="|", linestyle="None",
               markersize=12, markeredgewidth=2, label="Configured threshold"),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False, ncol=1,
              labelcolor="#afc0cd", fontsize=8.5)
    return fig


def extract_uploaded_scores(payload: bytes, bundle: dict) -> tuple[np.ndarray, np.ndarray, Image.Image]:
    """Run the existing shared preprocessing and feature path for one upload."""
    try:
        with Image.open(io.BytesIO(payload)) as uploaded:
            uploaded.load()
            preview = uploaded.convert("RGB").copy()
        with Image.open(io.BytesIO(payload)) as inference_image:
            gray = preprocess_image(inference_image)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("This file could not be read as a PNG or JPEG image. Try another X-ray file.") from exc
    features = extract_features(gray)[None, :]
    expected = int(bundle["model"].n_features_in_)
    if features.shape[1] != expected:
        raise ValueError(f"Feature size mismatch: expected {expected}, received {features.shape[1]}.")
    scores = positive_class_scores(bundle["model"], features)[0]
    return gray, scores, preview


if "upload_generation" not in st.session_state:
    st.session_state.upload_generation = 0
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "analysis_error" not in st.session_state:
    st.session_state.analysis_error = None


def clear_upload() -> None:
    st.session_state.upload_generation += 1
    st.session_state.analysis_result = None
    st.session_state.analysis_error = None


model_bundle = None
model_error = None
try:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Selected Random Forest model was not found: {MODEL_PATH}")
    model_bundle = load_selected_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
except Exception as exc:  # Keep the dashboard's status honest if artifact loading fails.
    model_error = str(exc)

model_ready = model_bundle is not None
model_size = MODEL_PATH.stat().st_size if model_ready else None
feature_count = int(model_bundle["model"].n_features_in_) if model_ready else None

metrics_data = None
metrics_available = False
if model_ready:
    try:
        with METRICS_PATH.open(encoding="utf-8") as metrics_file:
            candidate_metrics = json.load(metrics_file)
        metrics_available, _ = check_metrics_association(
            candidate_metrics, model_bundle, model_size
        )
        if metrics_available:
            metrics_data = candidate_metrics
    except (OSError, json.JSONDecodeError):
        metrics_available = False


st.markdown(
    f"""
    <div class="hero">
      <div class="brand">
        <div class="brand-mark" aria-hidden="true">
          <svg width="34" height="34" viewBox="0 0 40 40" fill="none">
            <path d="M19.8 8v23" stroke="#70e8dc" stroke-width="2.3" stroke-linecap="round"/>
            <path d="M18.8 18.2c-2.2-4.8-8.4-5.2-10.8-.7-1.6 3 .1 7.8 2.3 10.5 2.2 2.8 6.9 3.4 8.4.3" stroke="#4ed9cb" stroke-width="2.1" stroke-linecap="round"/>
            <path d="M20.9 18.2c2.2-4.8 8.4-5.2 10.8-.7 1.6 3-.1 7.8-2.3 10.5-2.2 2.8-6.9 3.4-8.4.3" stroke="#62cfff" stroke-width="2.1" stroke-linecap="round"/>
            <path d="M20 13.4h5.1l2.2 3.2h3.1" stroke="#b5fff1" stroke-width="1.5" stroke-linecap="round"/>
          </svg>
        </div>
        <div>
          <div class="brand-title">ThoraxIQ</div>
          <div class="brand-subtitle">Feature-Based Chest Radiograph Analysis</div>
        </div>
      </div>
      <div class="hero-right">
        <div class="status-pill"><span class="status-dot"></span>{'Random Forest Model Ready' if model_ready else 'Random Forest Model Unavailable'}</div>
        <div class="disclaimer-top">Educational research demonstration · Not for medical decisions</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if model_error:
    st.error(f"The selected Random Forest could not be loaded: {model_error}")
    st.stop()

st.markdown('<div class="eyebrow">System overview</div>', unsafe_allow_html=True)
st.markdown('<div class="section-copy">A compact view of the active model and its feature pipeline.</div>', unsafe_allow_html=True)

overview = st.columns(4)
overview[0].markdown(metric_card("Active algorithm", "Random Forest", "Multi-label · 14 outputs"), unsafe_allow_html=True)
overview[1].markdown(metric_card("Disease labels", "14", "Fixed label order"), unsafe_allow_html=True)
overview[2].markdown(metric_card("Input features", f"{feature_count:,}", "HOG · LBP · GLCM · 64 × 64 grayscale"), unsafe_allow_html=True)
overview[3].markdown(metric_card("Original RF artifact", f"{model_size / (1024 * 1024):.1f} MiB", "Loaded and feature-compatible"), unsafe_allow_html=True)

if metrics_available:
    st.markdown('<div style="height:.7rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Offline held-out evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Saved test-set metrics for this original Random Forest artifact. They are not confidence estimates for uploaded images.</div>', unsafe_allow_html=True)
    overview_metrics = st.columns(2)
    overview_metrics[0].markdown(metric_card("Test Macro F1", f"{metrics_data['macro_f1']:.4f}", "Equal weight per disease"), unsafe_allow_html=True)
    overview_metrics[1].markdown(metric_card("Test Micro F1", f"{metrics_data['micro_f1']:.4f}", "Aggregated label decisions"), unsafe_allow_html=True)

st.markdown('<div style="height:1.15rem"></div>', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">Image workspace</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Analyze a chest radiograph</div>', unsafe_allow_html=True)
st.markdown('<div class="section-copy">Upload a PNG or JPEG image. The file is processed in this app session and is not added to the training dataset.</div>', unsafe_allow_html=True)

with st.container(border=True):
    upload_columns = st.columns([.95, 1.05], gap="large")
    with upload_columns[0]:
        st.markdown('<div class="upload-hint">Drop an image below or choose a file from your device.<br>Supported formats: PNG, JPG, JPEG</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Chest X-ray image", type=SUPPORTED_TYPES,
            key=f"xray_upload_{st.session_state.upload_generation}",
            label_visibility="collapsed",
        )
        payload = uploaded_file.getvalue() if uploaded_file is not None else None
        upload_digest = sha256(payload).hexdigest() if payload is not None else None
        preview_image = None
        image_error = None
        if payload is not None:
            try:
                with Image.open(io.BytesIO(payload)) as preview_source:
                    preview_source.verify()
                with Image.open(io.BytesIO(payload)) as preview_source:
                    preview_image = preview_source.convert("RGB").copy()
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                image_error = "This file is not a readable PNG or JPEG image. Choose another file."
            st.markdown(
                f'<div class="file-meta"><b>File</b> · {escape(uploaded_file.name)}<br>'
                f'<b>Size</b> · {len(payload) / 1024:.1f} KB</div>',
                unsafe_allow_html=True,
            )
            if st.button("Remove image", key="remove_xray", width="stretch"):
                clear_upload()
                st.rerun()
        else:
            st.markdown('<div class="file-meta">No image selected yet.</div>', unsafe_allow_html=True)
    with upload_columns[1]:
        if preview_image is not None:
            st.image(preview_image, caption="Uploaded image preview", width="stretch")
        elif image_error:
            st.warning(image_error)
        else:
            st.markdown(
                '<div class="soft-panel" style="height:215px;display:flex;align-items:center;justify-content:center;text-align:center;color:#8fa9bb">'
                '<div><div style="font-size:2rem;color:#5ccfcf">◉</div><b>Image preview</b><br>'
                '<span style="font-size:.82rem">Your selected radiograph will appear here.</span></div></div>',
                unsafe_allow_html=True,
            )

    analyze = st.button(
        "Analyze X-ray", type="primary", width="stretch",
        disabled=payload is None or image_error is not None,
        key="analyze_xray",
    )

if payload is None:
    st.session_state.analysis_result = None
    st.session_state.analysis_error = None
elif st.session_state.analysis_result and st.session_state.analysis_result.get("upload_digest") != upload_digest:
    st.session_state.analysis_result = None
    st.session_state.analysis_error = None

if analyze and payload is not None:
    try:
        with st.spinner("Extracting image features and scoring 14 labels…"):
            gray, prediction_scores, _ = extract_uploaded_scores(payload, model_bundle)
        st.session_state.analysis_result = {
            "upload_digest": upload_digest,
            "scores": prediction_scores,
            "gray": gray,
        }
        st.session_state.analysis_error = None
    except Exception as exc:
        st.session_state.analysis_result = None
        st.session_state.analysis_error = str(exc)

if st.session_state.analysis_error:
    st.error(f"Unable to analyze this image: {st.session_state.analysis_error}")

result = st.session_state.analysis_result
if result and result.get("upload_digest") == upload_digest:
    st.markdown('<div style="height:1.4rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Model output</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prediction scores</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Scores are Random Forest vote fractions. They are not calibrated probabilities or diagnoses.</div>', unsafe_allow_html=True)
    threshold_values = np.asarray(model_bundle["thresholds"], dtype=float)
    score_values = np.asarray(result["scores"], dtype=float)
    predicted = score_values >= threshold_values
    result_columns = st.columns([.77, 1.23], gap="large")
    with result_columns[0]:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">Thresholded output</div>', unsafe_allow_html=True)
            if predicted.any():
                st.markdown("#### Labels above configured threshold")
                for index in np.flatnonzero(predicted):
                    st.markdown(
                        f"**{escape(DEFAULT_LABELS[index].replace('_', ' '))}**  "
                        f"<span style='color:#7de2d4'>score {score_values[index]:.3f}</span>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No disease label crossed its configured validation threshold. This result does not establish that an image is healthy.")
            st.caption("Thresholds are the saved per-disease values selected on validation data.")
            st.image(result["gray"], caption="Model input · grayscale, 64 × 64", width=220)
    with result_columns[1]:
        chart = make_score_chart(DEFAULT_LABELS, score_values, threshold_values)
        st.pyplot(chart, width="stretch")
        plt.close(chart)
    output_frame = pd.DataFrame({
        "Disease": [name.replace("_", " ") for name in DEFAULT_LABELS],
        "Prediction score": score_values,
        "Threshold": threshold_values,
        "At / above threshold": predicted,
    })
    st.dataframe(
        output_frame.style.format({"Prediction score": "{:.3f}", "Threshold": "{:.3f}"}),
        hide_index=True, width="stretch",
    )

st.markdown('<div style="height:1.3rem"></div>', unsafe_allow_html=True)
with st.expander("How this model works", expanded=False):
    st.markdown(
        """
        **1 · Image preparation.** The uploaded image is converted to grayscale and resized to 64 × 64 pixels with the same preprocessing function used during training.

        **2 · Handcrafted features.** The pipeline extracts 1,790 numeric descriptors: HOG summarizes local edge structure, LBP captures texture patterns, and GLCM summarizes grayscale relationships.

        **3 · Random Forest scoring.** A group of decision trees votes for each disease output. The displayed prediction score is the forest's positive-class vote fraction; it has not been verified as a calibrated probability.

        **4 · Multi-label outputs.** The model evaluates 14 disease labels independently, so more than one label can cross its saved validation threshold, or none can.

        **5 · Limits.** The dataset labels may be noisy and disease frequencies are imbalanced. This feature-based model is an educational machine learning experiment, not a clinically validated diagnostic system. Never use its output for medical decisions.
        """
    )

st.markdown(
    """
    <div class="footer">
      <div><b style="color:#c3d7e4">ThoraxIQ</b> · Active model: Random Forest</div>
      <div>Educational experiment only · Predictions are not medical advice and must not guide clinical decisions.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
