"""Soil Fertility Classification System - Streamlit interface.

Run with:  streamlit run app.py
The app only LOADS the trained model; it never retrains.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.prediction import ArtifactError, load_artifacts, predict_one  # noqa: E402
from src.validation import validate_input  # noqa: E402

APP_TITLE = "Soil Fertility Classification System"
HOME, INPUT, RESULTS, EVAL, ABOUT = ("Home", "Soil Sample Input", "Classification Results",
                                     "Model Evaluation", "About the System")
PAGES = [HOME, INPUT, RESULTS, EVAL, ABOUT]
DISCLAIMER = ("This tool supports preliminary soil fertility classification. It does not replace "
              "laboratory analysis or professional agronomic assessment.")

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.hero {background: linear-gradient(120deg, #2e7d32, #66a35a); color: #fff; padding: 1.6rem 1.8rem;
       border-radius: 14px; margin-bottom: 1.2rem;}
.hero h1 {color: #fff; margin: 0 0 .3rem 0; font-size: 2rem;}
.hero p {margin: 0; font-size: 1.05rem; opacity: .95;}
.result-card {border: 2px solid #2e7d32; border-radius: 14px; padding: 1.2rem 1.5rem;
              background: rgba(46,125,50,.08); text-align: center; margin: .5rem 0 1rem 0;}
.result-card .label {font-size: 2.4rem; font-weight: 700; color: #2e7d32;}
.result-card .sub {font-size: .95rem; opacity: .8;}
</style>
"""


@st.cache_resource(show_spinner=False)
def get_artifacts() -> dict:
    return load_artifacts()


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def widget_key(feature: str) -> str:
    return f"in_{feature}"


# ----------------------------------------------------------------- Home
def page_home(art: dict) -> None:
    st.markdown(f'<div class="hero"><h1>🌱 {APP_TITLE}</h1>'
                "<p>A machine learning decision support tool that classifies a soil sample into a "
                "fertility class from its measured laboratory parameters.</p></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("What is soil fertility classification?")
        st.write("Soil fertility is the ability of soil to supply nutrients and favourable conditions for plant growth. "
                 "Soil fertility classification categorises soils into fertility classes using selected soil "
                 "properties. This system learns the relationship between measured soil properties and the fertility "
                 "classes recorded in a laboratory-derived dataset, then applies it to a new sample.")
    with c2:
        st.subheader("How to use")
        st.markdown("1. Open **Soil Sample Input** in the sidebar.\n"
                    "2. Enter all soil measurements in the units shown.\n"
                    "3. Click **Classify Soil**.\n"
                    "4. Read the predicted class on **Classification Results**.\n"
                    "5. Click **Classify another sample** or **Reset** to start again.")
    st.subheader("Dataset and model information")
    meta = art.get("metadata")
    if not meta:
        st.info("Model metadata is not available.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Selected algorithm", meta["selected_algorithm"])
        b.metric("Samples used (after cleaning)", meta["rows_clean"])
        c.metric("Input parameters", len(meta["features"]))
        d.metric("Fertility classes", len(art["schema"]["class_names"]))
        st.caption(f"Classes: {', '.join(art['schema']['class_names'].values())}. "
                   f"Training rows: {meta['n_train']}; held-out test rows: {meta['n_test']}. "
                   f"Source: {meta['dataset_file']}.")
    st.warning(DISCLAIMER)


# ----------------------------------------------------------------- Input
def reset_inputs(schema: dict) -> None:
    for f in schema["features"]:
        st.session_state[widget_key(f)] = None
    st.session_state["input_errors"] = []
    st.session_state["result"] = None


def classify_callback(art: dict) -> None:
    schema = art["schema"]
    raw = {f: st.session_state.get(widget_key(f)) for f in schema["features"]}
    check = validate_input(raw, schema)
    st.session_state["input_errors"] = check.errors
    st.session_state["result"] = None
    if not check.ok:
        return
    try:
        pred = predict_one(art["model"], schema, check.values)
    except Exception as exc:  # noqa: BLE001
        st.session_state["input_errors"] = [f"The prediction could not be completed ({type(exc).__name__}). "
                                            "Please check the inputs and try again."]
        return
    st.session_state["result"] = {"values": check.values, "warnings": check.warnings, "prediction": pred}
    st.session_state["page"] = RESULTS


def page_input(art: dict) -> None:
    schema = art["schema"]
    st.header("Soil Sample Input")
    st.info("All 12 parameters are **required**: the trained model uses every one of them, and missing values "
            "are never filled in automatically. Enter values from a laboratory soil analysis.")
    for group, feats in config.FEATURE_GROUPS.items():
        st.subheader(group)
        cols = st.columns(3)
        for i, f in enumerate(feats):
            spec = schema["feature_info"][f]
            with cols[i % 3]:
                st.number_input(
                    f"{spec['label']} [{spec['unit']}] *", min_value=float(spec["hard_min"]),
                    max_value=float(spec["hard_max"]), value=None, step=float(spec["step"]),
                    format="%.2f", key=widget_key(f),
                    help=f"Required. Range in training data: {spec['observed_min']:g} to {spec['observed_max']:g} {spec['unit']}.")
    st.caption("* required. Units follow the dataset documentation reported in the literature; "
               "see About the System.")
    for msg in st.session_state.get("input_errors", []):
        st.error(msg)
    b1, b2, _ = st.columns([1, 1, 3])
    b1.button("Classify Soil", type="primary", on_click=classify_callback, args=(art,))
    b2.button("Reset", on_click=reset_inputs, args=(schema,))


# --------------------------------------------------------------- Results
def go_to_input() -> None:
    st.session_state["page"] = INPUT


def page_results(art: dict) -> None:
    schema, meta = art["schema"], art.get("metadata") or {}
    st.header("Classification Results")
    res = st.session_state.get("result")
    if not res:
        st.info("No sample has been classified yet. Enter soil measurements on the **Soil Sample Input** page.")
        st.button("Go to Soil Sample Input", on_click=go_to_input)
        return
    pred = res["prediction"]
    st.markdown(f'<div class="result-card"><div class="sub">Predicted soil fertility class</div>'
                f'<div class="label">{pred["label"]}</div></div>', unsafe_allow_html=True)
    for w in res["warnings"]:
        st.warning(w)
    st.subheader("Explanation")
    st.write(f"The trained **{meta.get('selected_algorithm', 'classifier')}** model assigned this sample to the "
             f"**{pred['label']}** fertility class, based on patterns learned from "
             f"{meta.get('n_train', 'the')} laboratory-analysed training samples. The class names are those of the "
             "dataset's fertility labels. The result is a data-driven, preliminary classification, not a laboratory "
             "or agronomic determination.")
    st.subheader("Input parameters used")
    medians = schema["class_medians_train"]
    rows = []
    for f in schema["features"]:
        row = {"Parameter": schema["feature_info"][f]["label"], "Unit": schema["feature_info"][f]["unit"],
               "Your value": res["values"][f]}
        for cls, vals in medians.items():
            row[f"{cls}-class median (training data)"] = round(vals[f], 3)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Class medians are shown for context only; the model does not classify by comparing to these medians.")
    if pred["probabilities"]:
        st.subheader("Model class probabilities")
        prob = pd.DataFrame({"Class": list(pred["probabilities"]), "Probability": list(pred["probabilities"].values())})
        st.bar_chart(prob.set_index("Class"))
        st.caption("These are the model's own estimates from the patterns it learned. They are not a guarantee "
                   "that the classification is correct. Tree-based models often give values close to 0 or 1.")
    st.subheader("Model information")
    st.write(f"Algorithm: **{meta.get('selected_algorithm', 'n/a')}**  |  Features used: {len(schema['features'])}")
    st.button("Classify another sample", on_click=go_to_input)
    st.warning(DISCLAIMER)


# ------------------------------------------------------------ Evaluation
def page_eval(art: dict) -> None:
    st.header("Model Evaluation")
    m = read_json(config.METRICS_PATH)
    if not m:
        st.error("Evaluation results were not found. Run: python -m src.evaluate_model")
        return
    st.write(f"Selected algorithm: **{m['selected_algorithm']}**. Metrics below were produced by running the saved "
             "model on a held-out test set that was never used for training or for model selection.")
    a, b, c, d = st.columns(4)
    a.metric("Test accuracy", f"{m['test_accuracy']:.3f}")
    b.metric("Precision (macro)", f"{m['test_precision_macro']:.3f}")
    c.metric("Recall (macro)", f"{m['test_recall_macro']:.3f}")
    d.metric("F1-score (macro)", f"{m['test_f1_macro']:.3f}")
    st.caption(f"Training accuracy is {m['train_accuracy']:.3f}. This is optimistic because the model has seen "
               "those rows; use the held-out test results to judge performance.")
    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        names = m["class_labels"]
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        ax.imshow(m["confusion_matrix"], cmap="Greens")
        ax.set_xticks(range(len(names)), names)
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        mx = max(max(r) for r in m["confusion_matrix"])
        for i, r in enumerate(m["confusion_matrix"]):
            for j, v in enumerate(r):
                ax.text(j, i, v, ha="center", va="center", color="white" if v > mx / 2 else "black")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Dataset size and class distribution")
        dist = pd.DataFrame(m["class_distribution"]).rename(columns={"train": "Training", "test": "Test"})
        dist.loc["Total"] = dist.sum()
        st.dataframe(dist)
        st.bar_chart(pd.DataFrame(m["class_distribution"]).rename(columns={"train": "Training", "test": "Test"}))
    st.subheader("Per-class performance (test set)")
    pc = pd.DataFrame(m["per_class"]).T.rename(columns={"precision": "Precision", "recall": "Recall",
                                                       "f1": "F1-score", "support": "Support"})
    st.dataframe(pc.round(3))
    if min(m["class_distribution"]["test"].values()) < 20:
        st.warning("One or more classes have very few test samples, so their per-class scores are statistically "
                   "unstable. Interpret them with caution.")
    st.subheader("Classification report")
    st.code(m["classification_report"])
    comp = config.COMPARISON_PATH
    if comp.exists():
        st.subheader("Comparison of candidate algorithms")
        df = pd.read_csv(comp)
        df["diagnostic_only"] = df["diagnostic_only"].map({True: "diagnostic", False: ""})
        st.dataframe(df.round(3).rename(columns={"diagnostic_only": "note"}), hide_index=True)
        st.caption("The model was selected by cross-validated macro-F1 on the training data only. "
                   "'Diagnostic' rows are reference baselines and are never selected.")


# ----------------------------------------------------------------- About
def page_about(art: dict) -> None:
    st.header("About the System")
    st.write("**Purpose.** To classify soil fertility from measured soil parameters using a machine learning model, "
             "as a decision support tool for preliminary assessment.")
    st.write("**Role of machine learning.** A classifier was trained on labelled soil samples. It learns which "
             "combinations of measurements are associated with each fertility class and applies that pattern to new "
             "samples. Several algorithms were compared, and the one shown on the evaluation page was selected.")
    st.subheader("Input parameters")
    info = art["schema"]["feature_info"]
    st.dataframe(pd.DataFrame([{"Parameter": v["label"], "Unit": v["unit"],
                                "Range in training data": f"{v['observed_min']:g} to {v['observed_max']:g}"}
                               for v in info.values()]), hide_index=True)
    st.subheader("Limitations")
    st.markdown(
        "- The model was trained on a single publicly available dataset that mainly represents one geographical "
        "region; it has not been validated on Nigerian soils.\n"
        "- The dataset documents neither the method used to assign its fertility labels nor the units of every "
        "column. The class names Low/Medium/High follow published studies that use this dataset.\n"
        "- The 'High' class has few samples, so predictions for it are the least reliable.\n"
        "- Inputs outside the training ranges are flagged; predictions for them are extrapolations.\n"
        "- The system classifies soil fertility only. It does not recommend crops, fertilizers or yields.")
    st.warning(DISCLAIMER)


# ------------------------------------------------------------------ Main
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🌱", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.title("🌱 Soil Fertility")
    try:
        art = get_artifacts()
    except ArtifactError as exc:
        st.error(str(exc))
        st.stop()
        return
    if "page" not in st.session_state:
        st.session_state["page"] = HOME
    page = st.sidebar.radio("Navigation", PAGES, key="page")
    st.sidebar.caption(DISCLAIMER)
    {HOME: page_home, INPUT: page_input, RESULTS: page_results, EVAL: page_eval, ABOUT: page_about}[page](art)


main()
