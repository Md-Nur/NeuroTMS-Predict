# explainability.py
import numpy as np
import lime
import lime.lime_tabular

def get_explainer(df, features):
    """
    Initializes a LIME explainer for the subject-level dataset.
    The explainer is trained on all 16 features (biomarkers + SD + TMS).
    """
    return lime.lime_tabular.LimeTabularExplainer(
        training_data=df[features].values,
        feature_names=features,
        class_names=['Low Risk', 'High Risk'],
        mode='classification',
        random_state=42
    )

def explain_risk(explainer, model, input_vector, feature_names):
    """
    Explains the baseline risk prediction (TMS = 0) for a single subject.
    Returns a list of dictionaries with features, weights, and rules.
    """
    # Ensure TMS is set to 0 for baseline risk
    tms_idx = feature_names.index("TMS")
    input_vector_tms0 = np.array(input_vector).copy()
    input_vector_tms0[tms_idx] = 0
    
    # Explain relative to Class 1 (High Risk)
    exp = explainer.explain_instance(
        data_row=input_vector_tms0,
        predict_fn=model.predict_proba,
        num_features=len(feature_names)
    )
    
    # LIME returns list of tuples: (rule_string, weight)
    rules = exp.as_list()
    
    results = []
    for rule, weight in rules:
        matched_feature = "Unknown"
        for name in feature_names:
            if name in rule:
                matched_feature = name
                break
        if matched_feature == "TMS":
            continue
        results.append({
            "feature": matched_feature,
            "weight": weight,
            "rule": rule
        })
        
    return results

def explain_tms_benefit(explainer, model, input_vector, feature_names):
    """
    Explains the TMS treatment benefit (Absolute Risk Reduction) by comparing
    LIME explanations under TMS = 0 (baseline) vs TMS = 1 (treated).
    
    Benefit is: P(High Risk | TMS=0) - P(High Risk | TMS=1)
    
    Thus, a feature contributes positively to the benefit if it increases risk when untreated
    and that risk is reduced when treated: net_weight = weight_tms0 - weight_tms1.
    """
    tms_idx = feature_names.index("TMS")
    
    # Create input vector with TMS = 0
    input_tms0 = np.array(input_vector).copy()
    input_tms0[tms_idx] = 0
    
    # Create input vector with TMS = 1
    input_tms1 = np.array(input_vector).copy()
    input_tms1[tms_idx] = 1
    
    # Explain both instances relative to Class 1 (High Risk)
    exp_tms0 = explainer.explain_instance(
        data_row=input_tms0,
        predict_fn=model.predict_proba,
        num_features=len(feature_names)
    )
    
    exp_tms1 = explainer.explain_instance(
        data_row=input_tms1,
        predict_fn=model.predict_proba,
        num_features=len(feature_names)
    )
    
    # Map feature index to weight
    weights_tms0 = {idx: weight for idx, weight in exp_tms0.as_map()[1]}
    weights_tms1 = {idx: weight for idx, weight in exp_tms1.as_map()[1]}
    
    # Get decision rules under untreated baseline for descriptions
    rules_tms0 = {r[0].split()[0]: r[0] for r in exp_tms0.as_list()}
    
    net_contributions = []
    
    for idx, feature_name in enumerate(feature_names):
        # We ignore TMS feature itself in the final feature contribution chart
        # because we are explaining the benefit of modifying TMS.
        if feature_name == "TMS":
            continue
            
        w0 = weights_tms0.get(idx, 0.0)
        w1 = weights_tms1.get(idx, 0.0)
        
        # Positive net weight means risk is lower with TMS: weight_tms0 - weight_tms1
        net_weight = w0 - w1
        
        rule_desc = rules_tms0.get(feature_name, f"{feature_name} profile")
        
        # Clean up rule description to reflect the biomarker name
        # (LIME generates strings like 'FastRipples_21 > 5.0')
        net_contributions.append({
            "feature": feature_name,
            "weight": net_weight,
            "rule": rule_desc
        })
        
    # Sort by absolute weight descending
    net_contributions = sorted(net_contributions, key=lambda x: abs(x["weight"]), reverse=True)
    return net_contributions
