"""
visualization.py
Plotting helpers for the project.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_odds_ratios(df, top_n=10, save_path=None):
    df = df.copy()
    
    # Ensure 'Feature' is the index
    if 'Feature' in df.columns:
        df = df.set_index('Feature')
    
    # Sort and select top_n features
    df = df.sort_values('Odds_Ratio', ascending=True)
    to_plot = df.tail(top_n)
    
    # Color coding based on Odds Ratio
    colors = ['tomato' if x > 1 else 'skyblue' for x in to_plot['Odds_Ratio']]
    
    # Plot setup
    plt.figure(figsize=(8, max(4, 0.3 * len(to_plot))))
    plt.barh(to_plot.index, to_plot['Odds_Ratio'], color=colors)
    plt.axvline(1, color='gray', linestyle='--', label='Odds Ratio = 1')
    
    # Axis labels and title
    plt.xlabel('Odds Ratio')
    plt.ylabel('Features')
    plt.title('Odds Ratios (Top Features)')
    
    # Legend for color meaning
    red_patch = mpatches.Patch(color='tomato', label='Increases Odds Ratio (> 1)')
    blue_patch = mpatches.Patch(color='skyblue', label='Decreases Odds Ratio (< 1)')
    plt.legend(handles=[red_patch, blue_patch], loc='lower right')
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# def plot_odds_ratios(df, top_n=10, save_path=None):
#     df = df.copy()
#     if 'Feature' in df.columns:
#         df = df.set_index('Feature')
#     df = df.sort_values('Odds_Ratio', ascending=True)
#     to_plot = df.tail(top_n)
#     colors = ['tomato' if x>1 else 'skyblue' for x in to_plot['Odds_Ratio']]
#     plt.figure(figsize=(8, max(4, 0.3*len(to_plot))))
#     plt.barh(to_plot.index, to_plot['Odds_Ratio'], color=colors)
#     plt.axvline(1, color='gray', linestyle='--')
#     plt.xlabel('Odds Ratio'); plt.title('Odds Ratios (Top features)')
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight')
#     plt.show()