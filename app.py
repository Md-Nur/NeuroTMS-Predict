# app.py
import os
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Import feature engineering and target definition from train_pipeline
from train_pipeline import compute_engineered_features, define_targets
# Import LIME explainability functions
from explainability import get_explainers, explain_risk, explain_tms_benefit

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
    lime_results = sorted(lime_results, key=lambda x: x['weight'])
    
    features = [r['rule'] for r in lime_results]
    weights = [r['weight'] for r in lime_results]
    
    if classification_type == "risk":
        # For Risk Model:
        # positive weight increases risk (bad outcome) -> Red
        # negative weight decreases risk (good outcome) -> Green
        colors = ['#ef4444' if w > 0 else '#22c55e' for w in weights]
    else:
        # For TMS Benefit:
        # positive weight increases probability of benefit (good treatment response) -> Blue/Green
        # negative weight decreases probability of benefit -> Grey/Red
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
    if not (os.path.exists("risk_model.pkl") and os.path.exists("tms_benefit_model.pkl") and os.path.exists("pipeline_metadata.json")):
        return None
        
    risk_model = joblib.load("risk_model.pkl")
    tms_benefit_model = joblib.load("tms_benefit_model.pkl")
    
    with open("pipeline_metadata.json", "r") as f:
        metadata = json.load(f)
        
    # Pre-load dataset to initialize LIME explainers
    df_raw = pd.read_excel("TBI_sleep_FR_spikes_TMS.xlsx")
    df = compute_engineered_features(df_raw)
    df = define_targets(df)
    
    # Initialize explainers
    explainer_s1, explainer_s2 = get_explainers(df, metadata['stage1_features'], metadata['stage2_features'])
    
    return {
        "risk_model": risk_model,
        "tms_benefit_model": tms_benefit_model,
        "metadata": metadata,
        "explainer_s1": explainer_s1,
        "explainer_s2": explainer_s2,
        "df": df
    }

assets = load_pipeline_assets()

if assets is None:
    st.error("⚠️ Pipeline models and metadata not found. Please run the training pipeline first using: `uv run python train_pipeline.py`")
    st.stop()

# Extract preloaded components
risk_model = assets["risk_model"]
tms_benefit_model = assets["tms_benefit_model"]
metadata = assets["metadata"]
explainer_s1 = assets["explainer_s1"]
explainer_s2 = assets["explainer_s2"]
feature_medians = metadata["feature_medians"]

# Sidebar Panel: Patient Early Biomarkers Input
st.sidebar.header("🔧 Early Biomarkers Input")
st.sidebar.write("Input patient physiological biomarkers below:")

FastRipples = st.sidebar.number_input(
    "Fast Ripples (Events/hr)", 
    min_value=0.0, 
    value=float(feature_medians["FastRipples"]),
    step=1.0,
    help="Fast ripples frequency measured in early post-injury phase."
)

Spikes = st.sidebar.number_input(
    "Spikes (Events/hr)", 
    min_value=0.0, 
    value=float(feature_medians["Spikes"]),
    step=1.0,
    help="Interictal spike frequency."
)

NREM = st.sidebar.slider(
    "NREM Sleep (%)", 
    min_value=0.0, 
    max_value=100.0, 
    value=float(feature_medians["NREM"]),
    help="NREM sleep duration as percentage of total recording time."
)

REM = st.sidebar.slider(
    "REM Sleep (%)", 
    min_value=0.0, 
    max_value=100.0, 
    value=float(feature_medians["REM"]),
    help="REM sleep duration as percentage of total recording time."
)

Wake = st.sidebar.slider(
    "Wake State (%)", 
    min_value=0.0, 
    max_value=100.0, 
    value=float(feature_medians["Sleep_Fragmentation"] * (NREM + REM) / 120.0), # Derived roughly
    help="Wakefulness percentage."
)

DPI = st.sidebar.number_input(
    "Days Post-Injury (DPI)", 
    min_value=0, 
    value=int(feature_medians["DPI"]),
    step=1,
    help="Days post-injury when biomarkers were collected."
)

# Automated feature engineering from user inputs
FR_Delta_Coupling = (FastRipples * NREM) / 100.0
Sleep_Fragmentation = (Wake / (NREM + REM + 1e-5)) * 100.0

st.sidebar.markdown("---")
st.sidebar.write("### Calculated Features")
st.sidebar.markdown(f"**FR-Delta Coupling:** `{FR_Delta_Coupling:.2f}`")
st.sidebar.markdown(f"**Sleep Fragmentation Index:** `{Sleep_Fragmentation:.2f}`")


# Main Dashboard Header
st.title("🧠 Epilepsy Risk Stratification & TMS Treatment response")
st.markdown("### Clinical Decision Support System using Early Biomarkers")
st.markdown("---")

# Layout: Split into tabs
tab_overview, tab_risk, tab_tms, tab_explain, tab_metrics = st.tabs([
    "📊 Pipeline Overview", 
    "🧠 Stage 1: Risk Stratification", 
    "⚡ Stage 2: TMS Response Analysis",
    "🔍 LIME Explainability",
    "📋 Model Metrics & Cohort Limitations"
])

# ----------------- Tab 1: Pipeline Overview -----------------
with tab_overview:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Re-framed Two-Stage Predictive Pipeline
        
        This decision support tool uses a **two-stage predictive workflow** designed to address clinical utility and mitigate overfitting on small cohorts:
        
        1. **Stage 1 — Epilepsy Risk Stratification Model**:
           - Analyzes early biomarkers (`Fast Ripples`, `Spike Frequency`, `NREM/REM Sleep percentage`, `FR-Delta Coupling`, and `Sleep Fragmentation`) to estimate the overall risk of epilepsy progression.
           - Outputs a **Risk Score (0–1)** representing the probability of developing a severe seizure phenotype (defined by high seizure severity, low latency, and low threshold).
        
        2. **Stage 2 — TMS Treatment Effect Model**:
           - Compares the expected outcomes under the TMS treatment group vs. the non-TMS control group.
           - Computes the estimated **TMS Treatment Benefit** (absolute risk reduction/improvement in seizure outcome) for this specific patient's risk stratum and biomarker profile.
           - Outputs a recommendation: **Likely benefit from TMS** or **Low benefit**.
        """)
    
    with col2:
        st.info("""
        **How to use this dashboard:**
        1. Adjust patient biomarkers in the left sidebar.
        2. View **Stage 1** tab to analyze the epilepsy risk profile.
        3. View **Stage 2** tab to see if the patient is expected to benefit from TMS.
        4. View **LIME Explainability** tab to inspect the physiological drivers behind both predictions.
        """)
        
        st.warning("""
        **Research Cohort Notice**:
        This model was trained on a small cohort of animals (n=~8 unique subjects with 180 longitudinal repeated measures). Validation was performed strictly using Leave-One-Subject-Out (LOSO) cross-validation to prevent data leakage.
        """)

# Run inference
# Prepare input vector for Stage 1
X1_input = [FastRipples, Spikes, NREM, REM, FR_Delta_Coupling, Sleep_Fragmentation, DPI]
prob_risk = risk_model.predict_proba([X1_input])[0][1]
pred_risk = risk_model.predict([X1_input])[0]

# Prepare input vector for Stage 2 (with TMS=1 and TMS=0)
# Stage 2 features: stage1_features + ["Risk_Score", "TMS"]
# Using the predicted probability from Stage 1 as the Risk_Score feature
X2_input_tms1 = X1_input + [prob_risk, 1]
X2_input_tms0 = X1_input + [prob_risk, 0]

prob_good_tms1 = tms_benefit_model.predict_proba([X2_input_tms1])[0][1]
prob_good_tms0 = tms_benefit_model.predict_proba([X2_input_tms0])[0][1]
tms_benefit = prob_good_tms1 - prob_good_tms0

# ----------------- Tab 2: Stage 1 - Risk Stratification -----------------
with tab_risk:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Stage 1: Epilepsy Progression Risk Stratification")
        st.write("Predicts the probability of severe epilepsy progression based on early physiological biomarkers.")
        
        # Risk Score Display
        st.metric(label="Predicted Epilepsy Progression Risk Score", value=f"{prob_risk:.2%}")
        
        # Risk status badge
        if prob_risk > 0.5:
            st.markdown('<div class="status-badge status-badge-high">⚠️ HIGH RISK PROFILE</div>', unsafe_allow_html=True)
            st.markdown("""
            * **Clinical Interpretation**: The patient's biomarkers strongly align with animals that exhibited accelerated epilepsy progression, including high seizure severity, rapid onset (low latency), and lower seizure thresholds.
            """)
        else:
            st.markdown('<div class="status-badge status-badge-low">✅ LOW RISK PROFILE</div>', unsafe_allow_html=True)
            st.markdown("""
            * **Clinical Interpretation**: The patient's biomarkers align with a milder epilepsy progression path, characterized by lower seizure severity and longer latencies.
            """)
            
    with col2:
        st.subheader("Risk Contribution Drivers")
        st.write("Top biomarker drivers contributing to the patient's epilepsy risk score:")
        # Generate LIME explanation for Stage 1
        lime_results_s1 = explain_risk(explainer_s1, risk_model, X1_input, metadata['stage1_features'])
        fig_s1 = plot_lime_explanation(lime_results_s1, "Local Biomarker Contribution to Epilepsy Risk", "risk")
        st.pyplot(fig_s1)

# ----------------- Tab 3: Stage 2 - TMS Response Analysis -----------------
with tab_tms:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Stage 2: Transcranial Magnetic Stimulation (TMS) Benefit Analysis")
        st.write("Compares outcomes with TMS vs. without TMS for the patient's specific risk stratum and biomarker profile.")
        
        # Display counterfactual probabilities
        st.write(f"**Probability of Good Outcome (No TMS)**: `{prob_good_tms0:.2%}`")
        st.write(f"**Probability of Good Outcome (With TMS)**: `{prob_good_tms1:.2%}`")
        
        # Net Benefit Score
        st.metric(label="Estimated Net TMS Treatment Benefit (ARR)", value=f"{tms_benefit:+.2%}")
        
        # Treatment response recommendation
        if tms_benefit > 0.05:
            st.markdown('<div class="status-badge status-badge-benefit">⚡ LIKELY TO BENEFIT FROM TMS</div>', unsafe_allow_html=True)
            st.markdown(f"""
            * **Clinical Recommendation**: Administering TMS is predicted to improve the likelihood of a good seizure outcome by **{tms_benefit:.1%}** (absolute increase in probability of high threshold, high latency, and low severity).
            """)
        else:
            st.markdown('<div class="status-badge status-badge-no-benefit">❌ LOW BENEFIT FROM TMS</div>', unsafe_allow_html=True)
            st.markdown(f"""
            * **Clinical Recommendation**: TMS shows a marginal predicted outcome difference (**{tms_benefit:.1%}**) for this biomarker profile. Alternative treatments or monitoring should be considered.
            """)
            
    with col2:
        st.subheader("TMS Response Drivers")
        st.write("Top feature interactions driving the predicted response to TMS:")
        
        # Generate LIME explanation for Stage 2 Treatment Benefit
        # Explain Stage 2 model by comparing explanations under TMS=1 vs TMS=0
        # stage2_features contains TMS at the end, so we pass the vector of features
        # where we will overwrite TMS during explanation.
        lime_results_s2 = explain_tms_benefit(explainer_s2, tms_benefit_model, X2_input_tms1, metadata['stage2_features'])
        fig_s2 = plot_lime_explanation(lime_results_s2, "Local Feature Contribution to TMS Response Benefit", "benefit")
        st.pyplot(fig_s2)

# ----------------- Tab 4: LIME Explainability -----------------
with tab_explain:
    st.subheader("🔍 LIME Explanations Panel")
    st.markdown("""
    LIME (Local Interpretable Model-agnostic Explanations) approximates the complex model locally using a simple linear model around this patient's specific biomarker profile.
    This reveals **how each physiological feature shifts the model's output** for this particular case.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("### Stage 1: Epilepsy Risk Drivers")
        for res in lime_results_s1:
            color = "🔴" if res['weight'] > 0 else "🟢"
            direction = "increases" if res['weight'] > 0 else "decreases"
            st.markdown(f"{color} **{res['rule']}** {direction} epilepsy risk score by **{abs(res['weight']):.2f}**")
            
    with col2:
        st.write("### Stage 2: TMS Responsiveness Drivers")
        for res in lime_results_s2:
            color = "🔵" if res['weight'] > 0 else "🔴"
            direction = "improves" if res['weight'] > 0 else "reduces"
            st.markdown(f"{color} **{res['rule']}** {direction} TMS treatment efficacy by **{abs(res['weight']):.2f}**")

# ----------------- Tab 5: Model Metrics & Cohort Limitations -----------------
with tab_metrics:
    st.subheader("📋 Model Validation Metrics & Clinical Limitations")
    
    st.markdown("""
    ### Acknowledgment of Sample Size Concerns & Overfitting Risk
    When modeling preclinical datasets, overfitting is a major risk due to repeated longitudinal measures on a small cohort of subjects.
    
    To address this concern and ensure publication-quality validation, the pipeline implements the following safeguards:
    
    1. **Subject-Level Data Splitting**:
       - We explicitly avoid standard row-level random train/test splits.
       - A random row-level split would leak different days post-injury (DPI) from the same subject into both the training and test sets, artificially inflating validation accuracy.
    2. **Leave-One-Subject-Out (LOSO) Cross-Validation**:
       - The validation results reported below are generated by leaving all records of one unique animal out per fold, training on the remaining animals, and testing on the held-out animal.
       - This strictly measures the model's generalizability to *new, unseen subjects*.
    3. **Model Regularization**:
       - Extremely conservative tree depths (`max_depth=2`) and high regularization parameter values (`reg_alpha=1.5`, `reg_lambda=1.5`) are enforced.
    """)
    
    # Display validation metrics loaded from metadata
    metrics = metadata["metrics"]
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### Stage 1: Risk Model Validation")
        st.write(f"- **Leave-One-Subject-Out (LOSO) CV Accuracy**: `{metrics['stage1_loso_accuracy']:.2%}`")
        st.write(f"- **LOSO CV ROC-AUC Score**: `{metrics['stage1_loso_auc']:.2%}`")
        
    with col2:
        st.markdown("#### Stage 2: TMS Benefit Model Validation")
        st.write(f"- **Leave-One-Subject-Out (LOSO) CV Accuracy**: `{metrics['stage2_loso_accuracy']:.2%}`")
        st.write(f"- **LOSO CV ROC-AUC Score**: `{metrics['stage2_loso_auc']:.2%}`")