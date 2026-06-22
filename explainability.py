# explainability.py
import numpy as np
import lime
import lime.lime_tabular

def get_explainers(df, stage1_features, stage2_features):
    """
    Initializes LIME explainers for Stage 1 (Risk Model) and Stage 2 (TMS Benefit/Outcome Model).
    """
    # Stage 1 Explainer
    # We pass the training data as a numpy array
    explainer_stage1 = lime.lime_tabular.LimeTabularExplainer(
        training_data=df[stage1_features].values,
        feature_names=stage1_features,
        class_names=['Low Risk', 'High Risk'],
        mode='classification',
        random_state=42
    )
    
    # Stage 2 Explainer
    explainer_stage2 = lime.lime_tabular.LimeTabularExplainer(
        training_data=df[stage2_features].values,
        feature_names=stage2_features,
        class_names=['Low Benefit/Outcome', 'Good Outcome'],
        mode='classification',
        random_state=42
    )
    
    return explainer_stage1, explainer_stage2

def explain_risk(explainer, model, input_vector, feature_names):
    """
    Explains the Stage 1 Risk prediction for a single patient input.
    Returns a list of tuples: (feature_name, contribution_weight, rule_description)
    """
    # input_vector should be a 1D numpy array or list of shape (n_features,)
    exp = explainer.explain_instance(
        data_row=np.array(input_vector),
        predict_fn=model.predict_proba,
        num_features=len(feature_names)
    )
    
    # LIME returns list of tuples: (feature_index, weight)
    # We map feature_index back to feature_name and get the decision rules
    map_exp = exp.as_map()[1]  # get explanations for class 1 (High Risk)
    
    results = []
    # Decision rules are in exp.as_list()
    rules = exp.as_list() # list of tuples: ('rule_string', weight)
    
    for rule, weight in rules:
        # Find which feature name is in the rule string
        matched_feature = "Unknown"
        for name in feature_names:
            if name in rule:
                matched_feature = name
                break
        results.append({
            "feature": matched_feature,
            "weight": weight,
            "rule": rule
        })
        
    return results

def explain_tms_benefit(explainer, model, input_vector_without_tms, feature_names):
    """
    Explains the TMS Treatment Benefit by comparing LIME explanations 
    under TMS = 1 and TMS = 0.
    Returns the net feature contributions to the TMS benefit.
    """
    # Find feature indices of TMS and Risk_Score
    tms_idx = feature_names.index("TMS")
    
    # Create input vector with TMS = 1
    input_tms1 = np.array(input_vector_without_tms).copy()
    input_tms1[tms_idx] = 1
    
    # Create input vector with TMS = 0
    input_tms0 = np.array(input_vector_without_tms).copy()
    input_tms0[tms_idx] = 0
    
    # Explain both instances relative to class 1 (Good Outcome)
    exp_tms1 = explainer.explain_instance(
        data_row=input_tms1,
        predict_fn=model.predict_proba,
        num_features=len(feature_names)
    )
    
    exp_tms0 = explainer.explain_instance(
        data_row=input_tms0,
        predict_fn=model.predict_proba,
        num_features=len(feature_names)
    )
    
    # Map index to weight
    weights_tms1 = {idx: weight for idx, weight in exp_tms1.as_map()[1]}
    weights_tms0 = {idx: weight for idx, weight in exp_tms0.as_map()[1]}
    
    # Calculate net benefit contribution: weight_tms1 - weight_tms0
    net_contributions = []
    
    # decision rules for displaying
    rules_tms1 = {r[0].split()[0]: r[0] for r in exp_tms1.as_list()}
    
    for idx, feature_name in enumerate(feature_names):
        # We ignore TMS feature itself in the final feature contribution chart
        # because we are explaining the effect of changing TMS
        if feature_name == "TMS":
            continue
            
        w1 = weights_tms1.get(idx, 0.0)
        w0 = weights_tms0.get(idx, 0.0)
        net_weight = w1 - w0
        
        # Get rule description
        rule_desc = rules_tms1.get(feature_name, f"{feature_name} profile")
        
        net_contributions.append({
            "feature": feature_name,
            "weight": net_weight,
            "rule": rule_desc
        })
        
    # Sort by absolute weight descending
    net_contributions = sorted(net_contributions, key=lambda x: abs(x["weight"]), reverse=True)
    return net_contributions
