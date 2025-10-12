"""
model_training.py
Train a Logistic Regression classifier and save the pipeline.
"""

import logging
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils import save_model, save_plot

sns.set(style='whitegrid')

# === Paths ===
PROCESSED = Path('../data/processed/cleaned_data.csv')
MODEL_OUT = Path('../models/logistic_regression_model.joblib')
FIG_OUT = Path('../reports/figures/confusion_matrix.png')

# === Logging setup ===
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


# === Load processed data ===
def load_processed(path=PROCESSED):
    """Load preprocessed CSV dataset."""
    logging.info(f"Loading processed data from {path}")
    return pd.read_csv(path)


# === Build preprocessor ===
def build_preprocessor(X):
    """
    Build preprocessing pipeline for numeric and categorical columns.
    Scales numeric features and one-hot encodes categorical ones.
    """
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])
    return preprocessor, numeric_cols, cat_cols


# === Train logistic regression pipeline ===
def train_pipeline(X_train, y_train):
    """Train logistic regression pipeline."""
    preprocessor, _, _ = build_preprocessor(X_train)

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(max_iter=2000))
    ])

    logging.info("Training logistic regression pipeline...")
    pipe.fit(X_train, y_train)
    logging.info("Training complete.")
    return pipe


# === Evaluate pipeline ===
def evaluate(pipe, X_train, y_train, X_test, y_test):
    """Evaluate pipeline performance and save confusion matrix."""
    logging.info("Evaluating model performance...")

    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    logging.info(f"Training Accuracy: {train_acc:.2f}")
    logging.info(f"Test Accuracy: {test_acc:.2f}")

    print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_test_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')

    save_plot(fig, FIG_OUT)
    logging.info(f"Confusion matrix saved to {FIG_OUT}")


# === Main workflow ===
def main():
    """Main training workflow."""
    if not PROCESSED.exists():
        logging.error("Processed data not found. Run data_preprocessing.py first.")
        return

    df = load_processed()
    target = 'DefectStatus' if 'DefectStatus' in df.columns else 'Defect'

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = train_pipeline(X_train, y_train)
    evaluate(pipe, X_train, y_train, X_test, y_test)
    save_model(pipe, MODEL_OUT)
    logging.info(f"Model saved to {MODEL_OUT}")


if __name__ == '__main__':
    main()
