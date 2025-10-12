"""
model_evaluation.py
Utilities for evaluating models and extracting feature importance.
"""
import pandas as pd
import numpy as np

def get_feature_names(pipe):
    return pipe.named_steps['preprocessor'].get_feature_names_out()

def get_model_step(pipe):
    """Detects whether the final step is 'classifier' or 'regressor'."""
    if 'classifier' in pipe.named_steps:
        return 'classifier'
    elif 'regressor' in pipe.named_steps:
        return 'regressor'
    else:
        raise KeyError("No valid model step found (expected 'classifier' or 'regressor').")


def get_coefficients(pipe):
    step = get_model_step(pipe)
    model = pipe.named_steps[step]
    return model.coef_[0] if hasattr(model, "coef_") else None

def build_importance_df(pipe, X_train):
    feature_names = get_feature_names(pipe)
    coefs = get_coefficients(pipe)
    df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
    df['Odds_Ratio'] = np.exp(df['Coefficient'])
    return df.sort_values('Odds_Ratio', ascending=False)