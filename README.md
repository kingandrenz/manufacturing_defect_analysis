# 🏭 Predicting Manufacturing Defects — Data Analysis and Modeling

## 📘 Project Overview

This project focuses on predicting manufacturing defects using machine learning models.  
The dataset — **Predicting Manufacturing Defects Dataset** — was obtained from Kaggle:  
🔗 [Predicting Manufacturing Defects Dataset on Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/predicting-manufacturing-defects-dataset/data)

The goal is to analyze patterns and build predictive models to classify products as **defective** or **non-defective**, ultimately improving production efficiency and quality control.

---

## 🧩 Dataset Description

**File Used:** `data/interim/cleaned_data.csv`  
**Model Summary:** `reports/model_summary.csv`

**Key Columns Include:**

- Multiple **features (Feature1–Feature17)** describing production characteristics.
- **Defect Status** — the target variable indicating if a product is defective (1) or not (0).

---

## ⚙️ Data Analysis Steps

### 1️⃣ Data Cleaning & Preparation

- Loaded the dataset and handled missing values.
- Standardized column names.
- Encoded categorical variables if necessary.
- Scaled numerical features for modeling.

### 2️⃣ Exploratory Data Analysis (EDA)

- Visualized defect rate distribution.
- Examined feature correlations and relationships with `Defect Status`.
- Identified the top influencing features.

### 3️⃣ Model Building

Two machine learning models were built and evaluated:

- **Logistic Regression (with balanced class weights)**
- **Random Forest Classifier**

Each model was tested using **train-test split** and **cross-validation** for reliable performance estimation.

---

## 📊 Model Evaluation

| Model               | Cross-Validation Accuracy | Test Accuracy | Remarks                                   |
| :------------------ | :-----------------------: | :-----------: | :---------------------------------------- |
| Logistic Regression |           ~0.68           |   **0.84**    | Strong generalization, interpretable      |
| Random Forest       |         **0.83**          |   **0.84**    | Slightly higher CV accuracy, more complex |

---

## ✅ Conclusion

- **Logistic Regression (C=0.2, balanced class weight)** performed with ~85% accuracy on the test set.
- **Random Forest** achieved similar accuracy but is less interpretable and more complex.
- **Top Influential Features:** _Feature1, Feature2, Feature3_ (based on coefficients and feature importances).
- Logistic Regression is **recommended for deployment** due to its simplicity, interpretability, and consistent performance.
- Further tuning and validation on new production data are advised before final deployment.

---

## 📈 BI Dashboard in Tableau

A Tableau dashboard was created to visualize key insights:

- **Defect Distribution** by Feature.
- **Feature Impact** (Feature1, Feature2, Feature3 vs Defect Status).
- **Interactive Filters** for exploring feature relationships.
- **Model Summary Sheet** to show feature importance.

### Dashboard Data Sources:

- `data/interim/cleaned_data.csv`
- `reports/model_summary.csv`

---

## 🧠 Key Learnings

- Feature scaling significantly improved model convergence.
- Logistic Regression offered high accuracy with strong interpretability.
- Random Forest captured complex relationships but required more resources.
- BI dashboards provide a practical interface for production quality insights.

---

## 💻 Tools & Libraries Used

- **Python:** pandas, numpy, scikit-learn, matplotlib, seaborn
- **Visualization:** Tableau Desktop / Tableau Public
- **IDE:** Jupyter Notebook

---

## 📁 Project Structure

```
├── data/
│   ├── raw/
│   ├── interim/
│   │   └── cleaned_data.csv
├── reports/
│   └── model_summary.csv
├── notebooks/
│   └── defect_prediction.ipynb
├── README.md
```

---

## 👨‍🔬 Author

**Anthony Kanu (Flexteck)**  
📧 [Kanuchibueze@gmail.com](mailto:Kanuchibueze@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/flexteck/)
