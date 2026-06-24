# app.py
import os
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Import feature engineering and target definition from train_pipeline
from train_pipeline import pivot_and_engineer_data, define_targets
# Import LIME explainability functions
from explainability import get_explainer, explain_risk, explain_tms_benefit

# Page configuration for a premium, wide look
st.set_page_config(
    page_title="Epilepsy Risk Stratification & TMS Response Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich visual aesthetics, vibrant gradients, and premium styling
st.markdown("""
<style>
    /* Primary brand colors and backgrounds */
    .reportview-container {
        background: #f8f9fa;
    }
    
    /* Title and headers */
    h1 {
        color: #1e293b;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    h2, h3 {
        color: #334155;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    
    /* Custom badge container */
    .status-badge {
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 16px;
        display: inline-block;
        margin-top: 10px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
    }
    
    .status-badge-high {
        background-color: #fee2e2;
        color: #ef4444;
        border: 1px solid #fecaca;
    }
    
    .status-badge-low {
        background-color: #dcfce7;
        color: #22c55e;
        border: 1px solid #bbf7d0;
    }
    
    .status-badge-benefit {
        background-color: #dbeafe;
        color: #2563eb;
        border: 1px solid #bfdbfe;
    }
    
    .status-badge-no-benefit {
        background-color: #f1f5f9;
        color: #64748b;
        border: 1px solid #e2e8f0;
    }

    /* Warning container styling */
    .warning-box {
        background-color: #fffbeb;
        color: #b45309;
        padding: 16px;
        border-radius: 8px;
        border-left: 5px solid #f59e0b;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to plot LIME explanations using Matplotlib
def plot_lime_explanation(lime_results, title, classification_type="risk"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Sort results by weight so largest impact is at the top
    # For risk, we show absolute impact
    # For benefit, we show the signed difference
    lime_results = sorted(lime_results, key=lambda x: abs(x['weight']))
    
    features = [r['rule'] for r in lime_results]
    weights = [r['weight'] for r in lime_results]
    
    if classification_type == "risk":
        # For Risk Model:
        # positive weight increases risk (bad outcome) -> Red
        # negative weight decreases risk (good outcome) -> Green
        colors = ['#ef4444' if w > 0 else '#22c55e' for w in weights]
    else:
        # For TMS Benefit:
        # positive weight increases risk reduction (benefit) -> Blue
        # negative weight reduces risk reduction -> Red
        colors = ['#2563eb' if w > 0 else '#ef4444' for w in weights]
        
    bars = ax.barh(features, weights, color=colors, height=0.6)
    ax.axvline(0, color='#94a3b8', linestyle='--', linewidth=1.0)
    
    # Stylize the chart for publication look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.tick_params(colors='#475569', labelsize=9)
    ax.set_title(title, fontsize=12, fontweight='bold', color='#1e293b', pad=15)
    ax.set_xlabel('Local Relative Feature Impact (Weight)', fontsize=10, color='#475569')
    ax.grid(axis='x', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    return fig

# Load models and pipeline metadata on startup
@st.cache_resource
def load_pipeline_assets():
    if not (os.path.exists("elastic_net_model.pkl") and os.path.exists("random_forest_model.pkl") and os.path.exists("pipeline_metadata.json")):
        return None
        
    en_model = joblib.load("elastic_net_model.pkl")
    rf_model = joblib.load("random_forest_model.pkl")
    
    with open("pipeline_metadata.json", "r") as f:
        metadata = json.load(f)
        
    # Pre-load dataset to initialize LIME explainer
    df_raw = pd.read_excel("TBI_sleep_FR_spikes_TMS.xlsx")
    df_wide = pivot_and_engineer_data(df_raw)
    df = define_targets(df_wide)
    
    # Initialize explainer
    explainer = get_explainer(df, metadata['features'])
    
    return {
        "en_model": en_model,
        "rf_model": rf_model,
        "metadata": metadata,
        "explainer": explainer,
        "df": df
    }

assets = load_pipeline_assets()

if assets is None:
    st.error("⚠️ Pipeline models and metadata not found. Please run the training pipeline first using: `.venv/bin/python train_pipeline.py`")
    st.stop()

# Extract preloaded components
en_model = assets["en_model"]
rf_model = assets["rf_model"]
metadata = assets["metadata"]
explainer = assets["explainer"]
feature_medians = metadata["feature_medians"]

# Sidebar Panel: Model & Subject Covariates
st.sidebar.header("⚙️ Configuration")
model_choice = st.sidebar.selectbox(
    "Select Classifier Model", 
    ["Elastic Net Logistic Regression", "Random Forest Classifier"]
)

# Choose active model
selected_model = en_model if model_choice == "Elastic Net Logistic Regression" else rf_model

SD_choice = st.sidebar.selectbox(
    "Subject Baseline State (Sleep Deprivation)",
    ["Normal Sleep Pattern", "Sleep Deprived (SD)"]
)
SD = 1 if SD_choice == "Sleep Deprived (SD)" else 0

st.sidebar.markdown("---")

# Sidebar: Section A (21 DPI Inputs)
st.sidebar.subheader("📅 Section A: 21 DPI Inputs")
FastRipples_21 = st.sidebar.number_input(
    "Fast Ripples (Events/hr) [21 DPI]", 
    min_value=0.0, 
    value=float(feature_medians["FastRipples_21"]),
    step=1.0,
    help="Fast ripples frequency measured at 21 days post-injury."
)

Spikes_21 = st.sidebar.number_input(
    "Interictal Spikes (Events/hr) [21 DPI]", 
    min_value=0.0, 
    value=float(feature_medians["Spikes_21"]),
    step=1.0,
    help="Spike frequency measured at 21 days post-injury."
)

NREM_21 = st.sidebar.slider(
    "NREM Sleep (%) [21 DPI]", 
    min_value=0.0, 
    max_value=100.0, 
    value=float(feature_medians["NREM_21"]),
    help="NREM sleep percentage at 21 DPI."
)

REM_21 = st.sidebar.slider(
    "REM Sleep (%) [21 DPI]", 
    min_value=0.0, 
    max_value=100.0, 
    value=float(feature_medians["REM_21"]),
    help="REM sleep percentage at 21 DPI."
)

Wake_21 = st.sidebar.slider(
    "Wake State (%) [21 DPI]", 
    min_value=0.0, 
    max_value=100.0, 
    value=float(feature_medians["Wake_21"]),
    help="Wake percentage at 21 DPI."
)

# Sidebar: Section B (28 DPI Inputs)
st.sidebar.subheader("📅 Section B: 28 DPI Inputs")
FastRipples_28 = st.sidebar.number_input(
    "Fast Ripples (Events/hr) [28 DPI]", 
    min_value=0.0, 
    value=float(feature_medians["FastRipples_28"]),
    step=1.0,
    help="Fast ripples frequency measured at 28 days post-injury."
)

Spikes_28 = st.sidebar.number_input(
    "Interictal Spikes (Events/hr) [28 DPI]", 
    min_value=0.0, 
    value=float(feature_medians["Spikes_28"]),
    step=1.0,
    help="Spike frequency measured at 28 days post-injury."
)

NREM_28 = st.sidebar.slider(
    "NREM Sleep (%) [28 DPI]", 
    min_value=0.0, 
    max_value=100.0, 
    value=float(feature_medians["NREM_28"]),
    help="NREM sleep percentage at 28 DPI."
)

REM_28 = st.sidebar.slider(
    "REM Sleep (%) [28 DPI]", 
    min_value=0.0, 
    max_value=100.0, 
    value=float(feature_medians["REM_28"]),
    help="REM sleep percentage at 28 DPI."
)

Wake_28 = st.sidebar.slider(
    "Wake State (%) [28 DPI]", 
    min_value=0.0, 
    max_value=100.0, 
    value=float(feature_medians["Wake_28"]),
    help="Wake percentage at 28 DPI."
)

# Automated feature engineering from inputs
Sleep_Fragmentation_21 = (Wake_21 / (NREM_21 + REM_21 + 1e-5)) * 100
FR_Delta_Coupling_21 = (FastRipples_21 * NREM_21) / 100.0

Sleep_Fragmentation_28 = (Wake_28 / (NREM_28 + REM_28 + 1e-5)) * 100
FR_Delta_Coupling_28 = (FastRipples_28 * NREM_28) / 100.0

# Sidebar engineered feature displays
st.sidebar.markdown("---")
st.sidebar.write("### Calculated Features")
st.sidebar.markdown(f"**Sleep Fragmentation (21 DPI):** `{Sleep_Fragmentation_21:.1f}%`")
st.sidebar.markdown(f"**FR-Delta Coupling (21 DPI):** `{FR_Delta_Coupling_21:.2f}`")
st.sidebar.markdown(f"**Sleep Fragmentation (28 DPI):** `{Sleep_Fragmentation_28:.1f}%`")
st.sidebar.markdown(f"**FR-Delta Coupling (28 DPI):** `{FR_Delta_Coupling_28:.2f}`")


# Main Dashboard Header
st.title("🧠 Epilepsy Risk Stratification & TMS Treatment Response")
st.markdown("### Preclinical Pre-Injury EEG Biomarker Decision Support System (Proof of Concept)")
st.markdown("---")

# Sleep sum validations
total_sleep_21 = NREM_21 + REM_21 + Wake_21
total_sleep_28 = NREM_28 + REM_28 + Wake_28
if abs(total_sleep_21 - 100.0) > 1e-2 or abs(total_sleep_28 - 100.0) > 1e-2:
    st.warning(
        f"⚠️ **Sleep Percentage Notice**: The sleep percentages for 21 DPI sum to **{total_sleep_21:.1f}%** "
        f"and for 28 DPI sum to **{total_sleep_28:.1f}%**. For physiological accuracy, "
        "sliders should sum to approximately 100%."
    )

# Layout: Split into tabs
tab_overview, tab_risk, tab_tms, tab_metrics = st.tabs([
    "📊 Pipeline Overview", 
    "🧠 Stage 1: Risk Stratification", 
    "⚡ Stage 2: TMS Response Analysis",
    "📋 Model Metrics & Validation Rigor"
])

# ----------------- Tab 1: Pipeline Overview -----------------
with tab_overview:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Causal Inference S-Learner Pipeline Redesign
        
        This decision support system implements a **causal inference framework** using a unified **S-Learner model** to address the limitations of small preclinical datasets ($N=60$ subjects) and repeated-measures leakage:
        
        1. **Subject-Level Wide Schema**:
           - All longitudinal measures are pivoted on the subject ID (`AnimalID`). Each row represents one subject.
           - `DPI` (Days Post-Injury) is completely **removed** as a feature, eliminating temporal bias and extrapolation errors.
           - Biomarkers from early timepoints (**21 DPI** and **28 DPI**) serve as inputs to predict late-stage outcomes (**60 DPI**).
        
        2. **Stage 1 — Baseline Risk Stratification**:
           - Predicts the counterfactual probability of the subject developing a high-severity seizure phenotype at 60 DPI if left untreated:
             $$\\text{Risk Score} = P(\\text{High Risk}_{60} = 1 \\mid \\text{Biomarkers},\\text{TMS}=0)$$
        
        3. **Stage 2 — Counterfactual TMS Benefit**:
           - Compares the predicted risk under treatment ($TMS=1$) vs. no treatment ($TMS=0$). The **TMS Treatment Benefit** is defined as the absolute risk reduction (ARR):
             $$\\text{TMS Benefit} = P(\\text{High Risk}_{60} = 1 \\mid \\text{Biomarkers},\\text{TMS}=0) - P(\\text{High Risk}_{60} = 1 \\mid \\text{Biomarkers},\\text{TMS}=1)$$
        """)
    
    with col2:
        st.info("""
        **How to use this dashboard:**
        1. Select your preferred classifier model in the sidebar.
        2. Select the subject baseline state (Sleep Deprived vs. Normal Sleep).
        3. Adjust early biomarkers at 21 DPI (Section A) and 28 DPI (Section B).
        4. Inspect **Stage 1** for epilepsy risk and **Stage 2** for counterfactual treatment response.
        """)
        
        st.warning("""
        **Methodology Paper Framing Notice**:
        This tool is intended for a preliminary Method Paper. It represents a proof-of-concept framework to demonstrate how machine learning can assess preclinical TMS outcomes. It is not approved for clinical diagnostic or treatment decisions in humans.
        """)

# Construct the model input vector (16 elements)
X_input = [
    FastRipples_21, Spikes_21, NREM_21, REM_21, Wake_21, Sleep_Fragmentation_21, FR_Delta_Coupling_21,
    FastRipples_28, Spikes_28, NREM_28, REM_28, Wake_28, Sleep_Fragmentation_28, FR_Delta_Coupling_28,
    SD,
    0 # Placeholder for TMS
]

# Compute counterfactuals
X_tms0 = X_input.copy()
X_tms0[-1] = 0  # TMS = 0

X_tms1 = X_input.copy()
X_tms1[-1] = 1  # TMS = 1

# Model Inference
prob_risk = selected_model.predict_proba([X_tms0])[0][1]
prob_treated = selected_model.predict_proba([X_tms1])[0][1]
tms_benefit = prob_risk - prob_treated  # Absolute Risk Reduction

# ----------------- Tab 2: Stage 1 - Risk Stratification -----------------
with tab_risk:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Stage 1: Epilepsy Progression Risk Stratification")
        st.write("Predicts the baseline probability of severe epilepsy progression at 60 DPI in the absence of treatment.")
        
        # Risk Score Metric
        st.metric(label="Predicted Baseline Epilepsy Risk (Untreated)", value=f"{prob_risk:.2%}")
        
        # Risk status badge (threshold = 0.5)
        if prob_risk > 0.5:
            st.markdown('<div class="status-badge status-badge-high">⚠️ HIGH RISK PROFILE</div>', unsafe_allow_html=True)
            st.markdown("""
            * **Interpretation**: The subject's early biomarker trajectories (21 to 28 DPI) strongly match animals that progressed to a severe epileptic phenotype at 60 DPI (low threshold, short latency, high severity).
            """)
        else:
            st.markdown('<div class="status-badge status-badge-low">✅ LOW RISK PROFILE</div>', unsafe_allow_html=True)
            st.markdown("""
            * **Interpretation**: The subject's early biomarker trajectories align with a mild seizure phenotype at 60 DPI.
            """)
            
    with col2:
        st.subheader("Risk Contribution Drivers")
        st.write("LIME local explanations: how early biomarkers shape the baseline risk score.")
        
        # Generate LIME explanation
        lime_results_risk = explain_risk(explainer, selected_model, X_input, metadata['features'])
        fig_risk = plot_lime_explanation(lime_results_risk, f"Biomarker Drivers of Baseline Epilepsy Risk ({model_choice})", "risk")
        st.pyplot(fig_risk)

# ----------------- Tab 3: Stage 2 - TMS Response Analysis -----------------
with tab_tms:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Stage 2: Counterfactual TMS Benefit Analysis")
        st.write("Evaluates the counterfactual treatment benefit by setting the treatment indicator to active vs. inactive.")
        
        # Counterfactual output display
        st.write(f"**Baseline Risk (No TMS)**: `{prob_risk:.2%}`")
        st.write(f"**Counterfactual Risk (With TMS)**: `{prob_treated:.2%}`")
        
        # Net Benefit Score (ARR)
        st.metric(label="Estimated Net TMS Treatment Benefit (ARR)", value=f"{tms_benefit:+.2%}")
        
        # Treatment response recommendation (Threshold at 5% risk reduction)
        if tms_benefit > 0.05:
            st.markdown('<div class="status-badge status-badge-benefit">⚡ LIKELY TO BENEFIT FROM TMS</div>', unsafe_allow_html=True)
            st.markdown(f"""
            * **Recommendation**: TMS is predicted to reduce the subject's probability of developing a severe epileptic phenotype by **{tms_benefit:.1%}**.
            """)
        else:
            st.markdown('<div class="status-badge status-badge-no-benefit">❌ LOW BENEFIT FROM TMS</div>', unsafe_allow_html=True)
            st.markdown(f"""
            * **Recommendation**: TMS treatment yields a marginal predicted change in risk (**{tms_benefit:.1%}** ARR). Alternative strategies or monitoring should be prioritized.
            """)
            
    with col2:
        st.subheader("TMS Response Drivers")
        st.write("LIME counterfactual local explanations: which biomarkers drive the risk reduction.")
        
        # Generate LIME explanation for Stage 2 Treatment Benefit
        lime_results_tms = explain_tms_benefit(explainer, selected_model, X_input, metadata['features'])
        fig_tms = plot_lime_explanation(lime_results_tms, f"Biomarker Drivers of TMS Treatment Benefit ({model_choice})", "benefit")
        st.pyplot(fig_tms)

# ----------------- Tab 4: Model Metrics & Validation Rigor -----------------
with tab_metrics:
    st.subheader("📋 Leave-One-Out Cross-Validation (LOOCV) & Validation Rigor")
    
    st.markdown("""
    ### Validation Safeguards Against Overfitting and Repeated Measures
    Preclinical datasets with small sample sizes are highly vulnerable to inflated validation scores. 
    This redesigned pipeline introduces key safeguards to ensure publication-grade validity:
    
    1. **Elimination of Repeated Measures**: By reshaping the dataset to a subject-level format, each animal constitutes exactly one independent row ($N=60$). This prevents temporal data leakage between rows.
    2. **Leave-One-Out Cross-Validation (LOOCV)**: In each fold, a model is trained on $N-1$ animals ($59$ subjects) and validated on the remaining $1$ subject. This strictly evaluates generalization to unseen animals.
    3. **Bootstrapped Confidence Intervals**: Standard point estimates do not reflect the uncertainty of small sample sizes. We report performance metrics with **95% Confidence Intervals** computed via 1000 bootstrap resamples of the LOOCV predictions.
    """)
    
    metrics = metadata["metrics"]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📈 Elastic Net Logistic Regression")
        m_en = metrics["elastic_net"]
        st.write(f"- **LOOCV Accuracy**: `{m_en['accuracy']['mean']:.2%}` (95% CI: `{m_en['accuracy']['lower']:.2%}` - `{m_en['accuracy']['upper']:.2%}`)")
        st.write(f"- **LOOCV ROC-AUC Score**: `{m_en['auc']['mean']:.2%}` (95% CI: `{m_en['auc']['lower']:.2%}` - `{m_en['auc']['upper']:.2%}`)")
        st.write(f"- **LOOCV F1-Score**: `{m_en['f1']['mean']:.2%}` (95% CI: `{m_en['f1']['lower']:.2%}` - `{m_en['f1']['upper']:.2%}`)")
        st.info("💡 **Elastic Net** acts as a linear baseline and features strong L1 regularization (`C=0.5`), forcing non-contributing biomarker coefficients to zero to perform automatic feature selection.")
        
    with col2:
        st.markdown("#### 🌳 Random Forest Classifier")
        m_rf = metrics["random_forest"]
        st.write(f"- **LOOCV Accuracy**: `{m_rf['accuracy']['mean']:.2%}` (95% CI: `{m_rf['accuracy']['lower']:.2%}` - `{m_rf['accuracy']['upper']:.2%}`)")
        st.write(f"- **LOOCV ROC-AUC Score**: `{m_rf['auc']['mean']:.2%}` (95% CI: `{m_rf['auc']['lower']:.2%}` - `{m_rf['auc']['upper']:.2%}`)")
        st.write(f"- **LOOCV F1-Score**: `{m_rf['f1']['mean']:.2%}` (95% CI: `{m_rf['f1']['lower']:.2%}` - `{m_rf['f1']['upper']:.2%}`)")
        st.info("💡 **Random Forest** captures potential non-linear biomarker interactions. Its tree depth is strictly limited (`max_depth=3`) to prevent it from memorizing the small training set.")