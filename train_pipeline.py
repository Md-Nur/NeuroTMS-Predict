# train_pipeline.py
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from xgboost import XGBClassifier

def compute_engineered_features(df):
    """
    Computes clinical biomarker features from raw data.
    """
    df = df.copy()
    
    # 1. Sleep Fragmentation
    # Ratio of Wake percentage to sleep percentage (NREM + REM)
    df['Sleep_Fragmentation'] = (df['Wake'] / (df['NREM'] + df['REM'] + 1e-5)) * 100
    
    # 2. FR-Delta Coupling
    # Proxy: Fast ripples occurring during NREM sleep (which is dominated by delta waves)
    df['FR_Delta_Coupling'] = (df['FastRipples'] * df['NREM']) / 100.0
    
    # 3. Treatment indicators
    # The Excel has a 'Group' column with names like 'TBI', 'TBI_TMS', 'SD_TBI', 'SD_TBI_TMS', 'SHAM'
    df["TBI"] = df["Group"].apply(lambda x: 1 if "TBI" in x else 0)
    df["SD"] = df["Group"].apply(lambda x: 1 if x.startswith("SD") else 0)
    df["TMS"] = df["Group"].apply(lambda x: 1 if "TMS" in x else 0)
    
    return df

def define_targets(df):
    """
    Defines composite clinical risk and outcome scores for the two stages.
    """
    df = df.copy()
    
    # Normalize seizure severity, latency, and threshold to 0-1
    def min_max_normalize(series):
        if series.max() == series.min():
            return series * 0.0
        return (series - series.min()) / (series.max() - series.min())
        
    threshold_norm = min_max_normalize(df['SeizureThreshold'])
    latency_norm = min_max_normalize(df['SeizureLatency'])
    severity_norm = min_max_normalize(df['SeizureSeverity'])
    
    # Stage 1 Target: Epilepsy progression risk score (0 to 1, higher is worse/higher risk)
    # High risk corresponds to: low threshold (1-norm), low latency (1-norm), high severity (norm)
    df['Risk_Score'] = ((1.0 - threshold_norm) + (1.0 - latency_norm) + severity_norm) / 3.0
    df['High_Risk'] = (df['Risk_Score'] > df['Risk_Score'].median()).astype(int)
    
    # Stage 2 Target: Seizure Outcome Score (0 to 1, higher is better/milder progression)
    # Good outcome corresponds to: high threshold (norm), high latency (norm), low severity (1-norm)
    df['Outcome_Score'] = (threshold_norm + latency_norm + (1.0 - severity_norm)) / 3.0
    df['Good_Outcome'] = (df['Outcome_Score'] > df['Outcome_Score'].median()).astype(int)
    
    return df

def run_pipeline():
    print("=== Loading and Preparing Data ===")
    xlsx_path = "TBI_sleep_FR_spikes_TMS.xlsx"
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Data file {xlsx_path} not found.")
        
    df_raw = pd.read_excel(xlsx_path)
    df = compute_engineered_features(df_raw)
    df = define_targets(df)
    
    # Features for Stage 1 (Risk Model)
    stage1_features = [
        "FastRipples", "Spikes", 
        "NREM", "REM", 
        "FR_Delta_Coupling", "Sleep_Fragmentation", 
        "DPI"
    ]
    
    # Features for Stage 2 (TMS Treatment Effect Model)
    # Includes Stage 1 predicted risk score (or true risk score during training for consistency)
    # plus the original features and the treatment indicator
    stage2_features = stage1_features + ["Risk_Score", "TMS"]
    
    groups = df["AnimalID"]
    
    print(f"Total samples: {len(df)}")
    print(f"Unique animals (subjects): {groups.nunique()}")
    print(f"Unique groups list: {groups.unique().tolist()}")
    
    # ------------------ STAGE 1: RISK MODEL ------------------
    print("\n=== Training Stage 1: Risk Model (Epilepsy Risk Stratification) ===")
    X1 = df[stage1_features]
    y1 = df["High_Risk"]
    
    # Conservative hyperparameters to prevent overfitting on small cohort
    clf1 = XGBClassifier(
        n_estimators=60,
        max_depth=2,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.5,
        reg_lambda=1.5,
        random_state=42,
        eval_metric="logloss"
    )
    
    # Leave-One-Subject-Out (LOSO) Cross-Validation
    logo = LeaveOneGroupOut()
    y1_preds = np.zeros(len(df))
    y1_probs = np.zeros(len(df))
    
    for train_idx, test_idx in logo.split(X1, y1, groups):
        X_tr, y_tr = X1.iloc[train_idx], y1.iloc[train_idx]
        X_te, y_te = X1.iloc[test_idx], y1.iloc[test_idx]
        
        fold_model = XGBClassifier(**clf1.get_params())
        fold_model.fit(X_tr, y_tr)
        
        y1_preds[test_idx] = fold_model.predict(X_te)
        y1_probs[test_idx] = fold_model.predict_proba(X_te)[:, 1]
        
    s1_acc = accuracy_score(y1, y1_preds)
    s1_auc = roc_auc_score(y1, y1_probs)
    s1_f1 = f1_score(y1, y1_preds)
    print(f"Stage 1 LOSO CV Accuracy: {s1_acc:.4f}")
    print(f"Stage 1 LOSO CV ROC-AUC:  {s1_auc:.4f}")
    print(f"Stage 1 LOSO CV F1-Score: {s1_f1:.4f}")
    
    # Train final Stage 1 model on all data
    clf1.fit(X1, y1)
    joblib.dump(clf1, "risk_model.pkl")
    print("Saved risk_model.pkl")
    
    # Add predicted risk score back to dataframe for Stage 2 training
    # Use final predictions
    df['Predicted_Risk_Score'] = clf1.predict_proba(X1)[:, 1]
    
    # ------------------ STAGE 2: TREATMENT EFFECT MODEL ------------------
    print("\n=== Training Stage 2: Treatment Effect Model (TMS Benefit) ===")
    
    # S-Learner approach: train on all data predicting Good_Outcome using features + TMS
    # Features include the predicted risk score from Stage 1
    X2 = df[stage1_features + ["Predicted_Risk_Score", "TMS"]].rename(
        columns={"Predicted_Risk_Score": "Risk_Score"}
    )
    y2 = df["Good_Outcome"]
    
    clf2 = XGBClassifier(
        n_estimators=60,
        max_depth=2,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.5,
        reg_lambda=1.5,
        random_state=42,
        eval_metric="logloss"
    )
    
    # LOSO CV for Stage 2
    y2_preds = np.zeros(len(df))
    y2_probs = np.zeros(len(df))
    
    for train_idx, test_idx in logo.split(X2, y2, groups):
        X_tr, y_tr = X2.iloc[train_idx], y2.iloc[train_idx]
        X_te, y_te = X2.iloc[test_idx], y2.iloc[test_idx]
        
        fold_model = XGBClassifier(**clf2.get_params())
        fold_model.fit(X_tr, y_tr)
        
        y2_preds[test_idx] = fold_model.predict(X_te)
        y2_probs[test_idx] = fold_model.predict_proba(X_te)[:, 1]
        
    s2_acc = accuracy_score(y2, y2_preds)
    s2_auc = roc_auc_score(y2, y2_probs)
    s2_f1 = f1_score(y2, y2_preds)
    print(f"Stage 2 LOSO CV Accuracy: {s2_acc:.4f}")
    print(f"Stage 2 LOSO CV ROC-AUC:  {s2_auc:.4f}")
    print(f"Stage 2 LOSO CV F1-Score: {s2_f1:.4f}")
    
    # Train final Stage 2 model on all data
    clf2.fit(X2, y2)
    joblib.dump(clf2, "tms_benefit_model.pkl")
    print("Saved tms_benefit_model.pkl")
    
    # Save pipeline metadata and baseline statistics
    # Median feature values are used for fallbacks in Streamlit inputs
    medians = df[stage1_features].median().to_dict()
    
    metadata = {
        "stage1_features": stage1_features,
        "stage2_features": list(X2.columns),
        "feature_medians": medians,
        "metrics": {
            "stage1_loso_accuracy": s1_acc,
            "stage1_loso_auc": s1_auc,
            "stage2_loso_accuracy": s2_acc,
            "stage2_loso_auc": s2_auc
        },
        "risk_threshold": float(df['Risk_Score'].median()),
        "outcome_threshold": float(df['Outcome_Score'].median())
    }
    
    with open("pipeline_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print("Saved pipeline_metadata.json")
    print("=== Pipeline Training Completed Successfully ===")

if __name__ == "__main__":
    run_pipeline()
