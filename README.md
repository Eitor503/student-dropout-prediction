# Student Dropout Prediction Using Logistic Regression

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://student-dropout-prediction-uew3jvqm4nlhhqbdgzfkxe.streamlit.app/)

> 🚀 **Live Demo:** Try the interactive web application directly in your browser without any installation: **[Launch Student Dropout Prediction App](https://student-dropout-prediction-uew3jvqm4nlhhqbdgzfkxe.streamlit.app/)**

An end-to-end Machine Learning project that predicts whether a student is likely to drop out or remain enrolled/graduate after their second academic period.

The project covers the entire ML lifecycle: exploratory data analysis (EDA), preprocessing, training a **Logistic Regression** pipeline, interpreting model coefficients, and deploying an interactive web interface built with **Streamlit**.


## 🎯 Project Overview & Objectives

* **Predictive Modeling:** Train a balanced binary classification model to identify students at risk of academic dropout.
* **Model Interpretability:** Analyze model coefficients to understand which academic, financial, and demographic factors influence predictions.
* **End-to-End Pipeline:** Demonstrate a modular machine learning workflow separating raw data, exploratory analysis, cleaning, training, and inference.
* **Deployment:** Provide an accessible, user-friendly Streamlit web application (hosted on Streamlit Community Cloud) for real-time inference and probability estimation.


## 📊 Dataset & Target Variable

This project uses the **"Predict Students' Dropout and Academic Success"** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/), consisting of 4,000+ student records and 40+ demographic, socioeconomic, and academic features.

### Dataset Source

The original dataset is available from:

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- [Kaggle](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention)

The raw dataset is **not included in this repository**. To reproduce the exploratory analysis and preprocessing steps, download the dataset and place the CSV file at:

```text
data/
└── raw/
    └── dataset.csv

```

To focus specifically on early risk detection, the original 3-class target was transformed into a binary outcome:

### Target Transformation

| Original Class | Target Mapping | Binary Class | Description |
| --- | --- | --- | --- |
| **Dropout** | `1` | **At Risk** | Student left the academic institution |
| **Enrolled** | `0` | **Retained** | Student remains active in the system |
| **Graduate** | `0` | **Retained** | Student completed their degree |

---

## 🏗️ Project Architecture

```text
Raw Data (UCI) ──► EDA & Preprocessing ──► Processed Dataset
                                                   │
Streamlit UI ◄── Inference API (predict.py) ◄── LogisticRegressionCV Pipeline

```

### 📁 Repository Structure

```text
.
├── app/
│   └── app.py                  # Streamlit frontend interface
├── data/
│   ├── raw/                    # Original UCI dataset
│   └── processed/              # Cleaned dataset & feature schemas
├── models/
│   └── logistic_regression_cv.pkl # Serialized trained pipeline
├── notebooks/                  # Step-by-step EDA, cleaning & modeling
├── reports/
│   └── figures/                # Saved visualizations and metrics
├── src/
│   ├── train.py                # Model training pipeline
│   └── predict.py              # Modular inference engine
├── requirements.txt
└── README.md

```

---

## ⚙️ Model Training & Class Imbalance

The core classifier is a Scikit-Learn pipeline consisting of a `StandardScaler` and `LogisticRegressionCV`.

Because dropout cases are critical to catch, class weights were balanced to penalize false negatives on the minority class (`Dropout`).

```python
LogisticRegressionCV(
    cv=5,
    random_state=42,
    max_iter=1000,
    n_jobs=-1,
    solver="lbfgs",
    class_weight="balanced"
)

```

### Key Metrics & Interpretability

* **Primary Optimization Metric:** **Recall** and **ROC-AUC** (prioritizing the detection of at-risk students over precision alone).
* **Overall Performance:** ~91% ROC-AUC across cross-validation folds.
* **Top Predictive Features:**
* 📉 **Strongest Retention Factor:** `Curricular units 2nd sem (approved)` (High negative coefficient).
* 📈 **Risk Factors:** Unpaid tuition fees, lower 1st/2nd semester course completions, and specific financial indicators.



---

## 🚀 Quick Start & Local Installation

If you prefer to run the project locally instead of using the [live web app](https://student-dropout-prediction-uew3jvqm4nlhhqbdgzfkxe.streamlit.app/):

### Prerequisites

* Python 3.9+
* `pip` package manager

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/Eitor503/student-dropout-prediction.git
cd student-dropout-prediction

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```

### 2. Run the Web Application

Launch the Streamlit interface locally:

```bash
streamlit run app/app.py

```

### 3. Retrain the Model (Optional)

To execute the training script and re-serialize the model pipeline:

```bash
python src/train.py

```

---

## 🖥️ Streamlit Application Interface

The Streamlit web app allows users to input individual student details (age, grades, fee status, scholarship status, etc.) and outputs:

1. **Predicted Status:** `Dropout` or `Graduate/Enrolled`.
2. **Probability Distribution:** Visual breakdown of predicted probabilities (e.g., 14.2% Dropout / 85.8% Retained).

---

## 🛠️ Tech Stack

* **Language:** Python
* **Data Handling:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Data Visualization:** Matplotlib, Seaborn
* **Interface & Deployment:** Streamlit, Streamlit Community Cloud
* **Environment:** Jupyter Notebooks, Git

---

## 📚 Citation

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021).
*Predict Students' Dropout and Academic Success* [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

Licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.

## ⚠️ Disclaimer & Limitations

* **Non-Causal Interpretations:** Feature coefficients show statistical relationships learned by the model and should not be interpreted as direct causal links.
* **Decision Support Only:** This tool is designed for experimental and educational purposes. It should serve as a supplementary indicator for academic counselors rather than an automated decision-maker.
