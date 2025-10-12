#!/usr/bin/env bash
set -e

# --- Define project root ---
ROOT="./manufacturing_defect_analysis"

echo "🧩 Creating project at ${ROOT}..."
rm -rf "${ROOT}"
mkdir -p   "${ROOT}/data/raw"   "${ROOT}/data/processed"   "${ROOT}/notebooks"   "${ROOT}/src"   "${ROOT}/models"   "${ROOT}/reports/figures"   "${ROOT}/references"

# --- Create README ---
cat > "${ROOT}/README.md" <<'README'
# Manufacturing Defect Analysis

This project analyzes manufacturing defect data to **classify** the likelihood or severity of defects using machine learning.

## 📁 Project Structure
- `data/` → raw & processed datasets  
- `src/` → modularized Python scripts  
- `notebooks/` → Jupyter notebooks for exploration and modeling  
- `models/` → trained ML models  
- `reports/` → visual outputs and evaluation metrics  
- `references/` → documentation and supporting material  

## 🚀 Quick Start
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the project setup**
   ```bash
   bash create_project.sh
   ```

3. **Start exploring**
   Open the notebooks in `notebooks/` to train and evaluate models.

## 🧠 Goal
Develop a robust classification pipeline to predict defect categories from manufacturing process features, improving quality control and reducing waste.

## 🧰 Technologies
- Python 3.x  
- pandas, numpy, matplotlib, seaborn  
- scikit-learn  
- Jupyter  

## 👤 Author
**Flextech (Anthony Kanu)**  
[LinkedIn](https://www.linkedin.com/in/flexteck/)
README

# --- Create Requirements File ---
cat > "${ROOT}/requirements.txt" <<'REQ'
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
REQ

# --- Create Example Notebook ---
cat > "${ROOT}/notebooks/defect_classification.ipynb" <<'NB'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Manufacturing Defect Classification\n",
    "\n",
    "This notebook explores and models manufacturing defect data using supervised learning."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
NB

# --- Create Example Python Module ---
cat > "${ROOT}/src/data_loader.py" <<'PY'
"""
data_loader.py
----------------
Utility functions for loading and preprocessing manufacturing defect datasets.
"""
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    return pd.read_csv(path)
PY

# --- Completion Message ---
echo "✅ Project structure created successfully at ${ROOT}"
