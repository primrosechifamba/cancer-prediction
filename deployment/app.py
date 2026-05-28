import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from model_loader import load_prediction_model
from preprocess import preprocess_image


CLASS_NAMES = ["A", "AGC", "ASC-H", "ASC-US", "HSIL", "LSIL", "NILM", "SC"]
CLASS_DESCRIPTIONS = {
    "A": "Adenocarcinoma",
    "AGC": "Atypical glandular cells",
    "ASC-H": "Atypical squamous cells, cannot exclude HSIL",
    "ASC-US": "Atypical squamous cells of undetermined significance",
    "HSIL": "High-grade squamous intraepithelial lesion",
    "LSIL": "Low-grade squamous intraepithelial lesion",
    "NILM": "Negative for intraepithelial lesion or malignancy",
    "SC": "Squamous carcinoma",
}
CLASS_COLORS = {
    "A": "#2563EB",
    "AGC": "#059669",
    "ASC-H": "#D97706",
    "ASC-US": "#DC2626",
    "HSIL": "#7C3AED",
    "LSIL": "#0891B2",
    "NILM": "#16A34A",
    "SC": "#374151",
}

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_model" / "best_model.keras"
METADATA_PATH = BASE_DIR / "models" / "best_model" / "metadata.json"
OVERALL_METRICS_PATH = BASE_DIR / "results" / "overall_metrics" / "all_overall_metrics.csv"
IMAGE_SIZE = 224


st.set_page_config(
    page_title="Cervical Cancer Prediction",
    page_icon=".",
    layout="wide",
)


def load_metadata():
    if METADATA_PATH.exists():
        with METADATA_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    return {}


@st.cache_resource
def get_model(model_path):
    return load_prediction_model(str(model_path))


@st.cache_resource
def get_legacy_best_model():
    return load_prediction_model(str(MODEL_PATH))


@st.cache_data
def load_overall_metrics():
    if OVERALL_METRICS_PATH.exists():
        df = pd.read_csv(OVERALL_METRICS_PATH)
        df["resolution"] = df["resolution"].astype(int)
        df["run"] = df["run"].astype(int)
        return df
    return pd.DataFrame()


def model_path_from_row(row):
    return BASE_DIR / "experiments" / row["model"] / str(int(row["resolution"])) / f"run_{int(row['run'])}" / "best_model.keras"


def get_default_model_rows(metrics_df):
    if metrics_df.empty:
        return pd.DataFrame()

    available = metrics_df.copy()
    available["model_path"] = available.apply(model_path_from_row, axis=1)
    available = available[available["model_path"].apply(lambda path: path.exists())]

    if available.empty:
        return available

    high_res = available[available["resolution"] == available["resolution"].max()]
    source = high_res if not high_res.empty else available
    return source.sort_values("f1_macro", ascending=False).head(3)


def preprocess_variants(image, image_size, use_tta):
    variants = [image]
    if use_tta:
        variants.extend(
            [
                image.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
                image.rotate(5, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255)),
                image.rotate(-5, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255)),
            ]
        )

    return np.vstack([preprocess_image(variant, image_size) for variant in variants])


def predict_with_ensemble(image, model_rows, use_tta=True):
    weighted_prediction = None
    total_weight = 0.0
    model_outputs = []

    for _, row in model_rows.iterrows():
        model_path = model_path_from_row(row)
        model = get_model(model_path)
        image_size = int(row["resolution"])
        batch = preprocess_variants(image, image_size, use_tta)
        probabilities = model.predict(batch, verbose=0).mean(axis=0)

        weight = float(row.get("f1_macro", row.get("f1_weighted", 1.0)))
        weighted_prediction = probabilities * weight if weighted_prediction is None else weighted_prediction + probabilities * weight
        total_weight += weight

        model_outputs.append(
            {
                "Model": row["model"],
                "Resolution": f"{int(row['resolution'])}x{int(row['resolution'])}",
                "Macro F1": round(float(row.get("f1_macro", 0)), 3),
                "Prediction": CLASS_NAMES[int(np.argmax(probabilities))],
                "Confidence": round(float(np.max(probabilities) * 100), 2),
            }
        )

    return weighted_prediction / total_weight, pd.DataFrame(model_outputs)


def inject_styles():
    st.markdown(
        """
        <style>
        :root {
            --ink: #172033;
            --muted: #64748B;
            --line: #D8E0EA;
            --paper: #FFFFFF;
            --field: #F6F8FB;
            --primary: #0F766E;
            --primary-dark: #134E4A;
            --accent: #2563EB;
            --button-bg: #0F766E;
            --button-bg-hover: #115E59;
            --button-text: #FFFFFF;
        }
        .stApp {
            background:
                radial-gradient(circle at 18% 0%, rgba(15, 118, 110, 0.16), transparent 30%),
                radial-gradient(circle at 85% 6%, rgba(37, 99, 235, 0.12), transparent 26%),
                linear-gradient(180deg, #EEF6F5 0%, #F7FAFC 250px, #F7FAFC 100%);
            color: var(--ink);
        }
        .stApp, .stApp p, .stApp span, .stApp label, .stApp div {
            color: #172033;
        }
        .main .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #102A43;
        }
        h1, h2, h3, p {
            letter-spacing: 0;
        }
        [data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"], [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] div {
            color: #172033;
        }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: var(--primary-dark);
        }
        [data-testid="stMarkdownContainer"] {
            color: #172033;
        }
        [data-testid="stMarkdownContainer"] p {
            color: #172033;
        }
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
            color: #526173;
        }
        [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {
            color: #172033;
        }
        .app-shell {
            border: 1px solid rgba(15, 118, 110, 0.18);
            background: linear-gradient(135deg, rgba(255,255,255,0.94), rgba(239,250,247,0.92));
            border-radius: 8px;
            padding: 22px 26px;
            box-shadow: 0 20px 48px rgba(15, 23, 42, 0.1);
            margin-bottom: 18px;
            position: relative;
            overflow: hidden;
        }
        .app-shell:after {
            content: "";
            position: absolute;
            right: -80px;
            top: -110px;
            width: 260px;
            height: 260px;
            background: radial-gradient(circle, rgba(15,118,110,0.16), transparent 68%);
            pointer-events: none;
        }
        .app-kicker {
            color: var(--primary);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        .app-title {
            margin: 0;
            font-size: 2.15rem;
            line-height: 1.1;
            color: #102A43;
            font-weight: 850;
        }
        .app-subtitle {
            margin: 8px 0 0 0;
            max-width: 820px;
            color: #526173;
            font-size: 1rem;
        }
        .status-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 14px;
        }
        .status-pill {
            border: 1px solid #CCE3DF;
            color: var(--primary-dark);
            background: #EFFAF7;
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 0.82rem;
            font-weight: 700;
        }
        .status-pill, .status-pill span, .status-pill div {
            color: var(--primary-dark);
        }
        .panel {
            background: var(--paper);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 19px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.065);
        }
        .panel, .panel div, .panel p, .panel span {
            color: #172033;
        }
        .panel-title {
            color: #102A43;
            font-size: 1.03rem;
            font-weight: 800;
            margin-bottom: 10px;
        }
        .diagnosis-card {
            border: 1px solid #CFE3E0;
            background: linear-gradient(180deg, #FFFFFF 0%, #F2FBF9 100%);
            border-left: 6px solid var(--primary);
            border-radius: 8px;
            padding: 20px 22px;
            margin-bottom: 14px;
        }
        .diagnosis-card, .diagnosis-card div, .diagnosis-card p {
            color: #172033;
        }
        .diagnosis-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        .diagnosis-class {
            font-size: 2rem;
            font-weight: 900;
            color: var(--primary-dark);
            margin: 0;
        }
        .diagnosis-desc {
            color: #526173;
            margin-top: 6px;
            font-size: 0.98rem;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin: 12px 0 14px 0;
        }
        .metric-card {
            background: linear-gradient(180deg, #FFFFFF, #F8FBFD);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px 15px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.9);
        }
        .metric-card, .metric-card div {
            color: #172033;
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 750;
            text-transform: uppercase;
        }
        .metric-value {
            color: #102A43;
            font-size: 1.38rem;
            font-weight: 850;
            margin-top: 4px;
        }
        .top-card {
            border: 1px solid var(--line);
            background: #FFFFFF;
            border-radius: 8px;
            padding: 13px 14px;
            margin-bottom: 9px;
        }
        .top-card, .top-card div, .top-card span {
            color: #172033;
        }
        .top-card-head {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            align-items: baseline;
        }
        .top-class {
            font-size: 1rem;
            font-weight: 850;
            color: #102A43;
        }
        .top-prob {
            font-weight: 850;
            color: var(--primary-dark);
        }
        .top-desc {
            margin-top: 3px;
            color: var(--muted);
            font-size: 0.84rem;
        }
        .bar-row {
            margin: 10px 0 12px 0;
        }
        .bar-head {
            display: flex;
            justify-content: space-between;
            font-size: 0.88rem;
            color: #263445;
            margin-bottom: 4px;
        }
        .track {
            height: 9px;
            background: #E8EEF5;
            border-radius: 999px;
            overflow: hidden;
        }
        .fill {
            height: 100%;
            border-radius: 999px;
        }
        .alert-box {
            border: 1px solid #F3D19C;
            background: #FFF8EB;
            color: #7C4A03;
            border-radius: 8px;
            padding: 12px 14px;
            font-size: 0.92rem;
            margin: 10px 0 14px 0;
        }
        .alert-box, .alert-box div {
            color: #7C4A03;
        }
        .empty-panel {
            border: 1px dashed #B8C7D9;
            background: #FFFFFF;
            border-radius: 8px;
            padding: 26px;
            color: #526173;
            text-align: center;
        }
        div[data-testid="stFileUploader"] section {
            border: 1px dashed #8BBDB7;
            border-radius: 8px;
            background: linear-gradient(180deg, #FFFFFF, #F7FCFB);
            padding: 10px;
        }
        div[data-testid="stFileUploader"] section, div[data-testid="stFileUploader"] section * {
            color: #172033;
        }
        button, [role="button"], .stButton button, .stDownloadButton button,
        div[data-testid="stFileUploader"] button {
            background: var(--button-bg) !important;
            color: var(--button-text) !important;
            border: 1px solid var(--button-bg) !important;
            border-radius: 8px !important;
            font-weight: 800 !important;
            box-shadow: 0 8px 18px rgba(15, 118, 110, 0.2) !important;
        }
        button *, [role="button"] *, .stButton button *, .stDownloadButton button *,
        div[data-testid="stFileUploader"] button * {
            color: var(--button-text) !important;
        }
        button:hover, [role="button"]:hover, .stButton button:hover, .stDownloadButton button:hover,
        div[data-testid="stFileUploader"] button:hover {
            background: var(--button-bg-hover) !important;
            border-color: var(--button-bg-hover) !important;
            color: var(--button-text) !important;
        }
        button:focus, [role="button"]:focus, .stButton button:focus,
        div[data-testid="stFileUploader"] button:focus {
            outline: 3px solid rgba(15, 118, 110, 0.22) !important;
            outline-offset: 2px !important;
        }
        [data-testid="stFileUploaderDropzone"] button {
            min-height: 38px;
            padding: 0 14px;
        }
        [data-testid="stFileUploaderDropzone"] small {
            color: #526173 !important;
        }
        [data-baseweb="tab-list"] {
            gap: 6px;
            border-bottom: 1px solid var(--line);
        }
        [data-baseweb="tab"] {
            background: #EEF3F7;
            border-radius: 8px 8px 0 0;
            color: #172033;
            padding: 8px 12px;
            border: 1px solid #D8E0EA;
            border-bottom: none;
        }
        [data-baseweb="tab"] p {
            color: #172033;
            font-weight: 750;
        }
        [aria-selected="true"] {
            background: #DFF3EF !important;
        }
        [aria-selected="true"] p {
            color: #0F4F49 !important;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }
        div[data-testid="stDataFrame"] * {
            color: #172033;
        }
        .stAlert, .stAlert div, .stAlert p {
            color: #172033;
        }
        @media (max-width: 800px) {
            .metric-grid {
                grid-template-columns: 1fr;
            }
            .app-title {
                font-size: 1.65rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def pct(value):
    return f"{value:.2f}%"


def render_header(model_count, mode_label):
    st.markdown(
        f"""
        <div class="app-shell">
            <div class="app-kicker">Cervical cytology classifier</div>
            <h1 class="app-title">Cervical Cancer Prediction</h1>
            <p class="app-subtitle">Deep learning inference dashboard for eight cervical cytology categories.</p>
            <div class="status-row">
                <span class="status-pill">{mode_label}</span>
                <span class="status-pill">{model_count} active model{'s' if model_count != 1 else ''}</span>
                <span class="status-pill">Top-3 review enabled</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel_start(title):
    st.markdown(f'<div class="panel"><div class="panel-title">{html.escape(title)}</div>', unsafe_allow_html=True)


def render_panel_end():
    st.markdown("</div>", unsafe_allow_html=True)


def render_prediction_card(predicted_class, confidence, description):
    color = CLASS_COLORS.get(predicted_class, "#0F766E")
    st.markdown(
        f"""
        <div class="diagnosis-card" style="border-left-color:{color};">
            <div class="diagnosis-label">Predicted category</div>
            <h2 class="diagnosis-class">{html.escape(predicted_class)}</h2>
            <div class="diagnosis-desc">{html.escape(description)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(confidence, margin, mode_label):
    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{pct(confidence)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Top-1 margin</div>
                <div class="metric-value">{pct(margin)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Mode</div>
                <div class="metric-value" style="font-size:1.05rem;">{html.escape(mode_label)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_cards(probability_df):
    for rank, (_, row) in enumerate(probability_df.head(3).iterrows(), start=1):
        class_name = row["Class"]
        probability = float(row["Probability"] * 100)
        color = CLASS_COLORS.get(class_name, "#2563EB")
        st.markdown(
            f"""
            <div class="top-card">
                <div class="top-card-head">
                    <div class="top-class">{rank}. {html.escape(class_name)}</div>
                    <div class="top-prob">{probability:.2f}%</div>
                </div>
                <div class="top-desc">{html.escape(row["Description"])}</div>
                <div class="track" style="margin-top:9px;">
                    <div class="fill" style="width:{probability:.3f}%; background:{color};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_probability_bars(probability_df):
    for _, row in probability_df.iterrows():
        class_name = row["Class"]
        probability = float(row["Probability"] * 100)
        color = CLASS_COLORS.get(class_name, "#2563EB")
        st.markdown(
            f"""
            <div class="bar-row">
                <div class="bar-head">
                    <span>{html.escape(class_name)}</span>
                    <strong>{probability:.2f}%</strong>
                </div>
                <div class="track">
                    <div class="fill" style="width:{probability:.3f}%; background:{color};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar(default_model_rows, metadata, use_ensemble, use_tta):
    st.header("Inference")
    st.caption("Configuration")
    st.write(f"Weighted ensemble: **{'On' if use_ensemble else 'Off'}**")
    st.write(f"Test-time augmentation: **{'On' if use_tta else 'Off'}**")

    st.divider()
    st.caption("Active models")

    if use_ensemble and not default_model_rows.empty:
        for _, row in default_model_rows.iterrows():
            st.markdown(
                f"**{row['model']}** `{int(row['resolution'])}x{int(row['resolution'])}`  \n"
                f"Macro F1 `{float(row.get('f1_macro', 0)):.3f}`"
            )
    elif metadata:
        st.markdown(
            f"**{metadata.get('model', 'N/A')}** `{metadata.get('resolution', IMAGE_SIZE)}x{metadata.get('resolution', IMAGE_SIZE)}`  \n"
            f"Macro F1 `{float(metadata.get('f1_macro', 0)):.3f}`"
        )
    else:
        st.info("No model metadata found.")

    st.divider()
    st.caption("Class key")
    for class_name in CLASS_NAMES:
        color = CLASS_COLORS.get(class_name, "#64748B")
        st.markdown(
            f'<span style="display:inline-block;width:10px;height:10px;background:{color};border-radius:50%;margin-right:7px;"></span>'
            f"**{class_name}** - {CLASS_DESCRIPTIONS[class_name]}",
            unsafe_allow_html=True,
        )


inject_styles()

metadata = load_metadata()
metrics_df = load_overall_metrics()
default_model_rows = get_default_model_rows(metrics_df)

with st.sidebar:
    use_ensemble = st.toggle("Weighted ensemble", value=not default_model_rows.empty)
    use_tta = st.toggle("Test-time augmentation", value=True)
    render_sidebar(default_model_rows, metadata, use_ensemble, use_tta)

active_model_count = len(default_model_rows) if use_ensemble and not default_model_rows.empty else 1
mode_label = "Weighted ensemble" if use_ensemble and not default_model_rows.empty else "Single model"
render_header(active_model_count, mode_label)

upload_col, result_col = st.columns([0.92, 1.18], gap="large")

with upload_col:
    render_panel_start("Specimen Image")
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name, width="stretch")
        st.caption(f"Image size: {image.size[0]} x {image.size[1]} px")
    else:
        image = None
        st.markdown(
            """
            <div class="empty-panel">
                Select a JPG or PNG cytology image.
            </div>
            """,
            unsafe_allow_html=True,
        )
    render_panel_end()

with result_col:
    render_panel_start("Prediction")

    if image is None:
        st.markdown(
            """
            <div class="empty-panel">
                Results will appear after image upload.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        if use_ensemble and not default_model_rows.empty:
            prediction_vector, model_outputs = predict_with_ensemble(image, default_model_rows, use_tta)
            prediction = np.expand_dims(prediction_vector, axis=0)
        else:
            model = get_legacy_best_model()
            image_size = int(metadata.get("resolution", IMAGE_SIZE)) if metadata else IMAGE_SIZE
            batch = preprocess_variants(image, image_size, use_tta)
            prediction = np.expand_dims(model.predict(batch, verbose=0).mean(axis=0), axis=0)
            model_outputs = pd.DataFrame()

        predicted_index = int(np.argmax(prediction))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(prediction[0][predicted_index] * 100)
        sorted_indices = np.argsort(prediction[0])[::-1]
        second_confidence = float(prediction[0][sorted_indices[1]] * 100)
        margin = confidence - second_confidence

        probability_df = pd.DataFrame(
            {
                "Class": CLASS_NAMES,
                "Description": [CLASS_DESCRIPTIONS[name] for name in CLASS_NAMES],
                "Probability": prediction[0],
            }
        ).sort_values("Probability", ascending=False)

        render_prediction_card(predicted_class, confidence, CLASS_DESCRIPTIONS[predicted_class])
        render_metrics(confidence, margin, mode_label)

        if confidence < 55 or margin < 10:
            st.markdown(
                """
                <div class="alert-box">
                    Low-confidence or close result. Review the top-3 classes and model agreement.
                </div>
                """,
                unsafe_allow_html=True,
            )

        tab_top, tab_distribution, tab_models = st.tabs(["Top-3", "Probabilities", "Model Agreement"])

        with tab_top:
            render_top_cards(probability_df)

        with tab_distribution:
            render_probability_bars(probability_df)
            st.dataframe(
                probability_df.assign(Probability=lambda df: (df["Probability"] * 100).round(2)),
                width="stretch",
                hide_index=True,
            )

        with tab_models:
            if not model_outputs.empty:
                st.dataframe(model_outputs, width="stretch", hide_index=True)
            else:
                st.info("Single-model mode is active.")

    render_panel_end()
