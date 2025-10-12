"""
visualization.py
Plotting helpers for the project.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_odds_ratios(df, top_n=10, save_path=None):
    df = df.copy()
    if 'Feature' in df.columns:
        df = df.set_index('Feature')
    df = df.sort_values('Odds_Ratio', ascending=True)
    to_plot = df.tail(top_n)
    colors = ['tomato' if x>1 else 'skyblue' for x in to_plot['Odds_Ratio']]
    plt.figure(figsize=(8, max(4, 0.3*len(to_plot))))
    plt.barh(to_plot.index, to_plot['Odds_Ratio'], color=colors)
    plt.axvline(1, color='gray', linestyle='--')
    plt.xlabel('Odds Ratio'); plt.title('Odds Ratios (Top features)')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()