"""
utils.py
Helper utilities for saving/loading models and plots.
"""
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

def save_model(model, path):
    """Save a trained model to the specified path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"[INFO] Model saved to: {path}")

def load_model(path):
    """Load a trained model from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    print(f"[INFO] Loaded model from: {path}")
    return joblib.load(path)

def save_plot(fig, path):
    """Save matplotlib figure to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Plot saved to: {path}")
