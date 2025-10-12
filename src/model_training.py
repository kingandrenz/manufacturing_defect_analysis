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
from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,  mean_squared_error, r2_score
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

# === Identify numeric and categorical columns ===
def split_numeric_categorical(X, threshold=20):
    """
    Identify numeric and categorical columns in the dataset.
    Automatically moves numeric columns with few unique values 
    (e.g., <= threshold) to categorical columns.
    """
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    print("Original Numeric:", numeric_cols)
    print("Original Categorical:", cat_cols)

    # --- Reclassify low-cardinality numeric columns ---
    for col in numeric_cols.copy():
        if X[col].nunique() <= threshold:
            cat_cols.append(col)
            numeric_cols.remove(col)

    # --- Print adjusted split ---
    print("\nAdjusted Numeric:", numeric_cols)
    print("Adjusted Categorical:", cat_cols)

    return numeric_cols, cat_cols
    

# === Build preprocessor ===
def build_preprocessor(X, numeric_cols, cat_cols):
    """
    Build preprocessing pipeline for numeric and categorical columns.
    Scales numeric features and one-hot encodes categorical ones.
    """

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])
    return preprocessor


# === Train pipeline ===
def train_pipeline(X_train, y_train, preprocessor, model, model_type='classification'):
    """
    Train a machine learning pipeline for classification or regression tasks.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series or np.array
        Training labels.
    preprocessor : sklearn.compose.ColumnTransformer
        Preprocessing pipeline for numeric and categorical columns.
    model : sklearn estimator
        The model instance to train (e.g., LogisticRegression(), RandomForestClassifier()).
    model_type : str, optional
        'classification' or 'regression' (for logging and clarity only).

    Returns
    -------
    pipe : sklearn.Pipeline
        Trained pipeline with preprocessing and model.
    """
    # Create the pipeline
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model) if model_type == 'classification' else ('regressor', model)
    ])

    model_name = model.__class__.__name__
    logging.info(f"Training {model_name} ({model_type}) pipeline...")

    # Train the pipeline
    pipe.fit(X_train, y_train)

    logging.info(f"{model_name} training complete.")

    return pipe


# === Evaluate pipeline ===
def evaluate(pipe, X_train, y_train, X_test, y_test, fig_out='/reports/figures/model_evaluation.png', show_plot=True):
    """
    Automatically evaluate a trained sklearn pipeline (classification or regression).

    Automatically detects model type from the final estimator.
    Displays and saves evaluation plots and metrics.

    Parameters
    ----------
    pipe : sklearn.Pipeline
        Trained pipeline object.
    X_train, y_train, X_test, y_test : pd.DataFrame / pd.Series
        Training and testing data.
    fig_out : str
        Path to save the plot.
    show_plot : bool
        Whether to display the figure in the notebook.
    """
    # --- Detect model type ---
    model = pipe.named_steps[list(pipe.named_steps.keys())[-1]] 
    task_type = 'classification' if is_classifier(model) else 'regression'

    logging.info(f"Detected model type: {task_type}")
    logging.info("Evaluating model performance...")

    # --- Predictions ---
    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)

    if task_type == 'classification':
        # --- Metrics ---
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        logging.info(f"Training Accuracy: {train_acc:.3f}")
        logging.info(f"Test Accuracy: {test_acc:.3f}")

        print(f"\nTraining Accuracy: {train_acc:.3f}")
        print(f"Test Accuracy: {test_acc:.3f}")
        print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

        # --- Confusion Matrix ---
        cm = confusion_matrix(y_test, y_test_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')

    else:
        # --- Regression Metrics ---
        train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
        test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
        r2 = r2_score(y_test, y_test_pred)

        logging.info(f"Train RMSE: {train_rmse:.3f}")
        logging.info(f"Test RMSE: {test_rmse:.3f}")
        logging.info(f"R² Score: {r2:.3f}")

        print(f"\nRegression Performance:")
        print(f"Train RMSE: {train_rmse:.3f}")
        print(f"Test RMSE: {test_rmse:.3f}")
        print(f"R²: {r2:.3f}")

        # --- Residual Plot ---
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.scatterplot(x=y_test, y=y_test - y_test_pred, alpha=0.6)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')

    # --- Save and Show ---
    fig.tight_layout()
    fig.savefig(fig_out)
    logging.info(f"Evaluation figure saved to {fig_out}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


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
