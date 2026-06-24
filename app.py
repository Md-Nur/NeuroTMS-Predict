# app.py
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Ensure the app's directory is in the python path for Streamlit Cloud imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page configuration for a premium, clean look
st.set_page_config(
    page_title="Epilepsy Progression & TMS Response Predictor",
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
        margin-bottom: 2px;
    }
    h2, h3 {
        color: #334155;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    
    /* Metrics container card styling */
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #1e293b;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 14px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Custom badge container */
    .status-badge {
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
        display: inline-block;
        margin-top: 5px;
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

    /* Info card styling */
    .info-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        color: #334155;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #cbd5e1;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

# Load model and pipeline metadata on startup
@st.cache_resource
def load_pipeline_assets():
    if not (os.path.exists("risk_model.pkl") and os.path.exists("pipeline_metadata.json")):
        return None
        
    model = joblib.load("risk_model.pkl")
    with open("pipeline_metadata.json", "r") as f:
        metadata = json.load(f)
        
    return {
        "model": model,
        "metadata": metadata
    }

assets = load_pipeline_assets()

if assets is None:
    st.error("⚠️ Pipeline model and metadata not found. Please run the training pipeline first using: `.venv/bin/python train_pipeline.py`")
    st.stop()

# Extract preloaded components
model = assets["model"]
metadata = assets["metadata"]
feature_medians = metadata["feature_medians"]

# Sidebar Panel: Inputs
st.sidebar.header("⚙️ Inputs & Biomarkers")

SD_choice = st.sidebar.selectbox(
    "Subject Baseline State (Sleep Deprivation)",
    ["Normal Sleep Pattern", "Sleep Deprived (SD)"]
)
SD = 1.0 if SD_choice == "Sleep Deprived (SD)" else 0.0

st.sidebar.markdown("---")

# Sidebar: Section A (21 DPI Inputs)
st.sidebar.subheader("📅 21 DPI Measurements")
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
    max_value=10.0, 
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
st.sidebar.subheader("📅 28 DPI Measurements")
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
    max_value=10.0, 
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
st.title("🧠 Epilepsy Progression & TMS Response Predictor")
st.markdown("##### Preclinical ML Decision Support System • Elastic Net Logistic Regression S-Learner")

# Sleep sum validations
total_sleep_21 = NREM_21 + REM_21 + Wake_21
total_sleep_28 = NREM_28 + REM_28 + Wake_28
if abs(total_sleep_21 - 100.0) > 1e-1 or abs(total_sleep_28 - 100.0) > 1e-1:
    st.warning(
        f"⚠️ **Sleep Percentage Notice**: The sleep percentages for 21 DPI sum to **{total_sleep_21:.1f}%** "
        f"and for 28 DPI sum to **{total_sleep_28:.1f}%**. For physiological accuracy, NREM, REM, and Wake "
        "should sum to approximately 100%."
    )

# Map UI inputs to standard feature vector (order must match model features)
input_dict = {
    "FastRipples_21": FastRipples_21,
    "Spikes_21": Spikes_21,
    "NREM_21": NREM_21,
    "REM_21": REM_21,
    "Wake_21": Wake_21,
    "Sleep_Fragmentation_21": Sleep_Fragmentation_21,
    "FR_Delta_Coupling_21": FR_Delta_Coupling_21,
    "FastRipples_28": FastRipples_28,
    "Spikes_28": Spikes_28,
    "NREM_28": NREM_28,
    "REM_28": REM_28,
    "Wake_28": Wake_28,
    "Sleep_Fragmentation_28": Sleep_Fragmentation_28,
    "FR_Delta_Coupling_28": FR_Delta_Coupling_28,
    "SD": SD
}

# Construct full 16-element vectors for model inference (15 biomarkers + TMS)
features_list = metadata["features"]
X_tms0 = [input_dict.get(f, 0.0) if f != "TMS" else 0.0 for f in features_list]
X_tms1 = [input_dict.get(f, 0.0) if f != "TMS" else 1.0 for f in features_list]

# Model Inference
prob_risk = model.predict_proba([X_tms0])[0][1]
prob_treated = model.predict_proba([X_tms1])[0][1]
tms_benefit = prob_risk - prob_treated  # Absolute Risk Reduction (ARR)

# ----------------- Main Section: Predictions & Outcomes -----------------
st.markdown("### 📊 Predicted Seizure Outcomes (60 DPI)")

# Renders three premium metric cards
col1, col2, col3 = st.columns(3)

with col1:
    badge_html = (
        '<div class="status-badge status-badge-high">⚠️ HIGH PROGRESSION RISK</div>' 
        if prob_risk > 0.5 else 
        '<div class="status-badge status-badge-low">✅ LOW PROGRESSION RISK</div>'
    )
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Baseline Seizure Risk (No TMS)</div>
        <div class="metric-value">{prob_risk:.2%}</div>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Counterfactual Risk (With TMS)</div>
        <div class="metric-value">{prob_treated:.2%}</div>
        <div style="font-size: 12px; color: #64748b; margin-top: 10px;">Predicted risk under TMS treatment protocol</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    benefit_badge_html = (
        '<div class="status-badge status-badge-benefit">⚡ LIKELY TO BENEFIT (ARR > 5%)</div>' 
        if tms_benefit > 0.05 else 
        '<div class="status-badge status-badge-no-benefit">❌ LOW TMS BENEFIT</div>'
    )
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Net Treatment Benefit (ARR)</div>
        <div class="metric-value">{tms_benefit:+.2%}</div>
        {benefit_badge_html}
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ----------------- Middle Section: Visualization & Explanation -----------------
col_chart1, col_chart2 = st.columns([3, 2])

with col_chart1:
    st.markdown("### 🧬 Biological Progression Drivers")
    st.markdown("This chart displays the exact global/local contribution of each biomarker to the baseline risk prediction. Only features selected by the Elastic Net model (non-zero coefficients) are shown.")
    
    # Calculate local contributions using exact model parameters
    coefs = metadata["coefficients"]
    means = metadata["scaler_mean"]
    scales = metadata["scaler_scale"]
    
    contribs = []
    for feat in metadata["biomarker_features"]:
        coef = coefs.get(feat, 0.0)
        # Skip features zeroed out by L1 regularization
        if abs(coef) < 1e-5:
            continue
            
        val = input_dict[feat]
        val_std = (val - means[feat]) / scales[feat]
        weight = coef * val_std
        
        # Clean up labels for the plot
        label = feat.replace("_21", " (21 DPI)").replace("_28", " (28 DPI)")
        label = label.replace("FastRipples", "Fast Ripples").replace("Spikes", "Interictal Spikes")
        label = label.replace("NREM", "NREM Sleep").replace("REM", "REM Sleep").replace("Wake", "Wake State")
        label = label.replace("Sleep_Fragmentation", "Sleep Frag.").replace("FR_Delta_Coupling", "FR-Delta Coupling")
        
        contribs.append({
            "name": label,
            "weight": weight
        })
        
    # Sort contributions by magnitude
    contribs = sorted(contribs, key=lambda x: abs(x['weight']))
    
    if len(contribs) > 0:
        names = [c['name'] for c in contribs]
        weights = [c['weight'] for c in contribs]
        
        # Color coding: red for risk-increasing, green for risk-reducing
        colors = ['#ef4444' if w > 0 else '#22c55e' for w in weights]
        
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(names, weights, color=colors, height=0.6)
        ax.axvline(0, color='#94a3b8', linestyle='--', linewidth=1.0)
        
        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        ax.tick_params(colors='#475569', labelsize=9)
        ax.set_xlabel('Local Relative Feature Impact (Coefficient × Standardized Value)', fontsize=10, color='#475569')
        ax.grid(axis='x', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No features selected by the model have a non-zero contribution.")

with col_chart2:
    st.markdown("### ⚡ Treatment Risk Comparison")
    st.markdown("Visual comparison of the subject's risk profile with vs. without TMS treatment.")
    
    # Simple risk comparison bar chart
    fig_comp, ax_comp = plt.subplots(figsize=(4.5, 4))
    scenarios = ['Baseline (No TMS)', 'Treated (With TMS)']
    probabilities = [prob_risk, prob_treated]
    
    colors_comp = ['#f87171' if prob_risk > 0.5 else '#34d399', '#60a5fa']
    
    bars = ax_comp.bar(scenarios, probabilities, color=colors_comp, width=0.5, edgecolor='#e2e8f0')
    
    # Add percentage labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax_comp.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.1%}", ha='center', va='bottom', fontweight='bold', color='#1e293b')
        
    ax_comp.set_ylim(0.0, 1.1)
    ax_comp.set_ylabel('Probability of Severe Epilepsy Progression', fontsize=9, color='#475569')
    ax_comp.spines['top'].set_visible(False)
    ax_comp.spines['right'].set_visible(False)
    ax_comp.spines['left'].set_color('#cbd5e1')
    ax_comp.spines['bottom'].set_color('#cbd5e1')
    ax_comp.tick_params(colors='#475569', labelsize=9)
    
    plt.tight_layout()
    st.pyplot(fig_comp)

st.markdown("---")

# ----------------- Bottom Section: Model Details & Cohort Stats -----------------
expander_stats = st.expander("📋 Model Performance Validation & Cohort Statistics", expanded=False)

with expander_stats:
    col_perf, col_cohort = st.columns(2)
    
    with col_perf:
        st.markdown("#### Model Performance (LOOCV)")
        st.write("Validation metrics evaluated using Leave-One-Out Cross-Validation (LOOCV) and 1,000 bootstrap resamples on a cohort of 60 independent subjects.")
        
        m = metadata["metrics"]["elastic_net"]
        
        st.markdown(f"- **LOOCV Accuracy**: `{m['accuracy']['mean']:.2%}` (95% CI: `{m['accuracy']['lower']:.2%}` - `{m['accuracy']['upper']:.2%}`)")
        st.markdown(f"- **LOOCV ROC-AUC**: `{m['auc']['mean']:.2%}` (95% CI: `{m['auc']['lower']:.2%}` - `{m['auc']['upper']:.2%}`)")
        st.markdown(f"- **LOOCV F1-Score**: `{m['f1']['mean']:.2%}` (95% CI: `{m['f1']['lower']:.2%}` - `{m['f1']['upper']:.2%}`)")
        
    with col_cohort:
        st.markdown("#### Average 60 DPI Seizure Metrics in Cohort")
        st.write("Observed average raw outcomes at 60 DPI in the experimental study cohort (N=60 total subjects):")
        
        stats = metadata["outcome_stats"]
        df_stats = pd.DataFrame({
            "Treatment Group": ["Untreated (TMS = 0)", "Treated (TMS = 1)"],
            "Seizure Severity Score (0-3)": [stats["SeizureSeverity_60"]["0"], stats["SeizureSeverity_60"]["1"]],
            "Seizure Threshold (mA)": [stats["SeizureThreshold_60"]["0"], stats["SeizureThreshold_60"]["1"]],
            "Seizure Latency (s)": [stats["SeizureLatency_60"]["0"], stats["SeizureLatency_60"]["1"]]
        })
        st.dataframe(df_stats.set_index("Treatment Group"), use_container_width=True)

st.info("💡 *Proof-of-Concept Decision Support Tool for preclinical mouse EEG research. Not approved for human medical diagnosis.*")