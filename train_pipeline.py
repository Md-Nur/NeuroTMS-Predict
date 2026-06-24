import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.base import clone

def pivot_and_engineer_data(df_raw):
    """
    Pivots longitudinal dataset to subject-level and computes engineered features.
    """
    # Split by DPI
    df_21 = df_raw[df_raw['DPI'] == 21].copy()
    df_28 = df_raw[df_raw['DPI'] == 28].copy()
    df_60 = df_raw[df_raw['DPI'] == 60].copy()
    
    # We keep AnimalID, Group, TMS, and biological variables for 21 DPI
    df_wide = df_21[['AnimalID', 'Group', 'TMS', 'FastRipples', 'Spikes', 'NREM', 'REM', 'Wake']].rename(
        columns={
            'FastRipples': 'FastRipples_21',
            'Spikes': 'Spikes_21',
            'NREM': 'NREM_21',
            'REM': 'REM_21',
            'Wake': 'Wake_21'
        }
    )
    
    # Extract biological variables for 28 DPI
    df_28_sub = df_28[['AnimalID', 'FastRipples', 'Spikes', 'NREM', 'REM', 'Wake']].rename(
        columns={
            'FastRipples': 'FastRipples_28',
            'Spikes': 'Spikes_28',
            'NREM': 'NREM_28',
            'REM': 'REM_28',
            'Wake': 'Wake_28'
        }
    )
    
    # Merge 21 and 28 DPI data
    df_wide = df_wide.merge(df_28_sub, on='AnimalID', how='inner')
    
    # Extract 60 DPI outcome measures
    df_60_sub = df_60[['AnimalID', 'SeizureThreshold', 'SeizureLatency', 'SeizureSeverity']].rename(
        columns={
            'SeizureThreshold': 'SeizureThreshold_60',
            'SeizureLatency': 'SeizureLatency_60',
            'SeizureSeverity': 'SeizureSeverity_60'
        }
    )
    
    # Merge 60 DPI targets
    df_wide = df_wide.merge(df_60_sub, on='AnimalID', how='inner')
    
    # Compute Sleep Deprivation (SD) covariate
    df_wide['SD'] = df_wide['Group'].apply(lambda x: 1 if "SD" in str(x) else 0)
    
    # Engineered Features at 21 DPI
    df_wide['Sleep_Fragmentation_21'] = (df_wide['Wake_21'] / (df_wide['NREM_21'] + df_wide['REM_21'] + 1e-5)) * 100
    df_wide['FR_Delta_Coupling_21'] = (df_wide['FastRipples_21'] * df_wide['NREM_21']) / 100.0
    
    # Engineered Features at 28 DPI
    df_wide['Sleep_Fragmentation_28'] = (df_wide['Wake_28'] / (df_wide['NREM_28'] + df_wide['REM_28'] + 1e-5)) * 100
    df_wide['FR_Delta_Coupling_28'] = (df_wide['FastRipples_28'] * df_wide['NREM_28']) / 100.0
    
    return df_wide

def define_targets(df):
    """
    Defines composite outcome score and risk classes at 60 DPI.
    """
    df = df.copy()
    
    # Helper to min-max normalize
    def min_max_normalize(series):
        if series.max() == series.min():
            return series * 0.0
        return (series - series.min()) / (series.max() - series.min())
        
    threshold_norm = min_max_normalize(df['SeizureThreshold_60'])
    latency_norm = min_max_normalize(df['SeizureLatency_60'])
    severity_norm = min_max_normalize(df['SeizureSeverity_60'])
    
    # 60 DPI Composite Epilepsy Severity Score (Risk Score, 0 to 1, higher is worse)
    df['Risk_Score_60'] = ((1.0 - threshold_norm) + (1.0 - latency_norm) + severity_norm) / 3.0
    
    # Binarize outcome via median split (Balanced classes)
    df['High_Risk_60'] = (df['Risk_Score_60'] > df['Risk_Score_60'].median()).astype(int)
    
    # Save raw normalized components for reference
    df['norm_threshold_60'] = threshold_norm
    df['norm_latency_60'] = latency_norm
    df['norm_severity_60'] = severity_norm
    
    return df

def compute_bootstrap_ci(y_true, y_probs, y_preds, n_bootstrap=1000, ci=95):
    """
    Computes 95% Confidence Intervals for Accuracy, AUC, and F1-score using Bootstrapping.
    """
    accs, aucs, f1s = [], [], []
    n_samples = len(y_true)
    
    # Set seed for reproducibility of bootstrap
    np.random.seed(42)
    
    for _ in range(n_bootstrap):
        # Sample indices with replacement
        indices = resample(np.arange(n_samples))
        
        # Check if bootstrap sample has both classes (needed for AUC)
        if len(np.unique(y_true[indices])) < 2:
            continue
            
        accs.append(accuracy_score(y_true[indices], y_preds[indices]))
        aucs.append(roc_auc_score(y_true[indices], y_probs[indices]))
        f1s.append(f1_score(y_true[indices], y_preds[indices]))
        
    lower_pct = (100 - ci) / 2.0
    upper_pct = 100 - lower_pct
    
    return {
        "accuracy": {
            "mean": float(np.mean(accs)),
            "lower": float(np.percentile(accs, lower_pct)),
            "upper": float(np.percentile(accs, upper_pct))
        },
        "auc": {
            "mean": float(np.mean(aucs)),
            "lower": float(np.percentile(aucs, lower_pct)),
            "upper": float(np.percentile(aucs, upper_pct))
        },
        "f1": {
            "mean": float(np.mean(f1s)),
            "lower": float(np.percentile(f1s, lower_pct)),
            "upper": float(np.percentile(f1s, upper_pct))
        }
    }

def run_loocv(X, y, model):
    """
    Executes Leave-One-Out Cross-Validation.
    """
    loo = LeaveOneOut()
    y_preds = np.zeros(len(y))
    y_probs = np.zeros(len(y))
    
    for train_idx, test_idx in loo.split(X):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        
        # Clone the model pipeline to reset fit
        fold_model = clone(model)
        fold_model.fit(X_tr, y_tr)
        
        y_preds[test_idx] = fold_model.predict(X_te)
        y_probs[test_idx] = fold_model.predict_proba(X_te)[:, 1]
        
    return y_probs, y_preds

def run_pipeline():
    print("=== Loading and Pivoting Data ===")
    xlsx_path = "TBI_sleep_FR_spikes_TMS.xlsx"
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Data file {xlsx_path} not found.")
        
    df_raw = pd.read_excel(xlsx_path)
    df_wide = pivot_and_engineer_data(df_raw)
    df = define_targets(df_wide)
    
    # Feature columns
    biomarker_features = [
        "FastRipples_21", "Spikes_21", "NREM_21", "REM_21", "Wake_21", "Sleep_Fragmentation_21", "FR_Delta_Coupling_21",
        "FastRipples_28", "Spikes_28", "NREM_28", "REM_28", "Wake_28", "Sleep_Fragmentation_28", "FR_Delta_Coupling_28",
        "SD"
    ]
    model_features = biomarker_features + ["TMS"]
    
    X = df[model_features]
    y = df["High_Risk_60"]
    
    print(f"Pivoted subject-level dataset shape: {df.shape}")
    print(f"Total unique animals: {df['AnimalID'].nunique()}")
    print(f"Experimental group distribution:\n{df['Group'].value_counts()}")
    print(f"High risk target distribution: {y.value_counts().to_dict()}")
    
    # ------------------ DEFINE INTERPRETABLE MODELS ------------------
    
    # Model 1: Elastic Net Logistic Regression (S-Learner)
    # saga solver is required for l1+l2 elasticnet penalty. 
    # C=0.5 provides strong L1 regularization to encourage feature sparsity (selection).
    clf_en = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            penalty='elasticnet', 
            solver='saga', 
            l1_ratio=0.5, 
            C=0.5, 
            random_state=42, 
            max_iter=10000
        ))
    ])
    
    # Model 2: Random Forest Classifier (S-Learner)
    # Restricted tree depth to prevent overfitting on small cohort
    clf_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        min_samples_leaf=3,
        random_state=42
    )
    
    # ------------------ LEAVE-ONE-OUT VALIDATION ------------------
    print("\n=== Validating Elastic Net Model (LOOCV) ===")
    en_probs, en_preds = run_loocv(X, y, clf_en)
    en_ci = compute_bootstrap_ci(y.values, en_probs, en_preds)
    print(f"Accuracy: {en_ci['accuracy']['mean']:.4f} (95% CI: {en_ci['accuracy']['lower']:.4f} - {en_ci['accuracy']['upper']:.4f})")
    print(f"ROC-AUC:  {en_ci['auc']['mean']:.4f} (95% CI: {en_ci['auc']['lower']:.4f} - {en_ci['auc']['upper']:.4f})")
    print(f"F1-Score: {en_ci['f1']['mean']:.4f} (95% CI: {en_ci['f1']['lower']:.4f} - {en_ci['f1']['upper']:.4f})")
    
    print("\n=== Validating Random Forest Model (LOOCV) ===")
    rf_probs, rf_preds = run_loocv(X, y, clf_rf)
    rf_ci = compute_bootstrap_ci(y.values, rf_probs, rf_preds)
    print(f"Accuracy: {rf_ci['accuracy']['mean']:.4f} (95% CI: {rf_ci['accuracy']['lower']:.4f} - {rf_ci['accuracy']['upper']:.4f})")
    print(f"ROC-AUC:  {rf_ci['auc']['mean']:.4f} (95% CI: {rf_ci['auc']['lower']:.4f} - {rf_ci['auc']['upper']:.4f})")
    print(f"F1-Score: {rf_ci['f1']['mean']:.4f} (95% CI: {rf_ci['f1']['lower']:.4f} - {rf_ci['f1']['upper']:.4f})")
    
    # ------------------ FIT FINAL MODELS ------------------
    print("\n=== Fitting Final Models on Complete Dataset ===")
    clf_en.fit(X, y)
    clf_rf.fit(X, y)
    
    # Save models
    joblib.dump(clf_en, "elastic_net_model.pkl")
    joblib.dump(clf_rf, "random_forest_model.pkl")
    
    # For backward compatibility, save the best-performing model as risk_model.pkl
    # (We compare LOOCV AUCs to decide)
    best_model_name = "Elastic Net" if en_ci['auc']['mean'] >= rf_ci['auc']['mean'] else "Random Forest"
    best_model = clf_en if best_model_name == "Elastic Net" else clf_rf
    joblib.dump(best_model, "risk_model.pkl")
    print(f"Saved {best_model_name} as the default 'risk_model.pkl'")
    print("Saved elastic_net_model.pkl and random_forest_model.pkl")
    
    # ------------------ SAVE METADATA ------------------
    # Medians for default slider inputs
    feature_medians = df[biomarker_features].median().to_dict()
    
    # Compute baseline outcome statistics
    # Average 60 DPI outcome values for untreated (TMS=0) vs treated (TMS=1)
    outcome_stats = df.groupby('TMS')[['SeizureThreshold_60', 'SeizureLatency_60', 'SeizureSeverity_60']].mean().to_dict()
    
    metadata = {
        "features": model_features,
        "biomarker_features": biomarker_features,
        "feature_medians": feature_medians,
        "metrics": {
            "elastic_net": {
                "accuracy": en_ci['accuracy'],
                "auc": en_ci['auc'],
                "f1": en_ci['f1']
            },
            "random_forest": {
                "accuracy": rf_ci['accuracy'],
                "auc": rf_ci['auc'],
                "f1": rf_ci['f1']
            }
        },
        "best_model_name": best_model_name,
        "outcome_stats": outcome_stats,
        "cohort_size": int(len(df))
    }
    
    with open("pipeline_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print("Saved pipeline_metadata.json")
    print("=== Pipeline Training Completed Successfully ===")

if __name__ == "__main__":
    run_pipeline()
