"""
model_evaluation.py
Utilities for evaluating models and extracting feature importance.
"""
import pandas as pd
import numpy as np

def get_feature_names(pipe):
    return pipe.named_steps['preprocessor'].get_feature_names_out()

def get_coefficients(pipe):
    coefs = pipe.named_steps['clf'].coef_[0]
    return coefs

def build_importance_df(pipe, X_train):
    feature_names = get_feature_names(pipe)
    coefs = get_coefficients(pipe)
    df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
    df['Odds_Ratio'] = np.exp(df['Coefficient'])
    return df.sort_values('Odds_Ratio', ascending=False)