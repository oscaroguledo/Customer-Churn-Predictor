# Customer Churn Predictor

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)
![Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626)
![License](https://img.shields.io/badge/License-MIT-green)

A supervised machine-learning study that predicts whether a subscription
service customer will cancel, based on their contract, service mix, tenure,
and billing profile. The full workflow — data acquisition, cleaning,
exploratory analysis, modelling, and evaluation — lives in a single
reproducible notebook, [`data.ipynb`](data.ipynb).

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Tech Stack](#tech-stack)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

Customer retention is materially cheaper than acquisition, so identifying
at-risk accounts before they churn is a high-value modelling problem. This
project frames churn as a binary classification task and compares three
baseline models, deliberately reporting **precision, recall, F1, and
ROC-AUC** rather than raw accuracy, which is misleading on an imbalanced
target (only ~26.5% of customers churn).

The notebook is self-contained: it downloads the dataset on first run, and
every cell executes top to bottom without manual intervention.

## Dataset

**IBM Telco Customer Churn** — 7,043 customers, 21 columns.

| Property | Value |
| --- | --- |
| Rows | 7,043 |
| Raw features | 19 (after dropping `customerID` and the target) |
| Features after one-hot encoding | 30 |
| Target | `Churn` (`Yes` / `No` → `1` / `0`) |
| Positive class rate | 26.5% |
| Source | [`IBM/telco-customer-churn-on-icp4d`](https://github.com/IBM/telco-customer-churn-on-icp4d) (raw CSV, fetched at runtime) |

The dataset is downloaded into an untracked `data/` directory the first
time the notebook runs, so no manual download or Kaggle credentials are
required.

## Methodology

1. **Environment setup & data loading** — fetch the raw CSV and load it
   into a pandas DataFrame.
2. **EDA & preprocessing**
   - Coerce `TotalCharges` from text to numeric; the 11 blank values
     belong to customers with zero tenure and are set to `0`.
   - Drop `customerID` (no predictive value).
   - Encode the target to `0` / `1`.
   - One-hot encode categorical features with `drop_first=True`.
   - Visualise the class imbalance to justify a stratified split.
3. **Splitting & scaling** — an 80/20 stratified train/test split, with
   `StandardScaler` fitted **only** on the training fold and applied to
   both, so no test-set statistics leak into training.
4. **Model training** — three baseline classifiers on identical inputs:
   Logistic Regression, Decision Tree, and Random Forest.
5. **Evaluation & interpretation** — a metrics table, ROC curves, and
   confusion matrices, followed by a churn-driver analysis using Logistic
   Regression coefficients and Random Forest feature importances.

## Results

Test-set performance (1,409 held-out customers, positive class = churn):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| **Logistic Regression** | **0.806** | **0.659** | **0.559** | **0.605** | **0.842** |
| Random Forest | 0.789 | 0.629 | 0.503 | 0.559 | 0.824 |
| Decision Tree | 0.725 | 0.482 | 0.479 | 0.481 | 0.646 |

Logistic Regression is the strongest and most interpretable baseline. The
untuned Decision Tree overfits and trails badly on ROC-AUC.

## Key Findings

**Signals that increase churn risk**

- Month-to-month contracts
- Fiber-optic internet service
- Payment by electronic check
- Paperless billing

**Signals that increase retention**

- Longer tenure
- One- and two-year contracts
- Higher total charges accumulated over time

These align with the intuition that low-commitment, high-friction billing
relationships churn fastest.

## Project Structure

```
Customer-Churn-Predictor/
├── data.ipynb          # End-to-end analysis notebook (5 sections)
├── requirements.txt     # Runtime and test dependencies
├── data/                # Downloaded dataset (git-ignored, created on first run)
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9 or newer
- An internet connection for the first run (dataset download)

### Installation

```bash
git clone https://github.com/oscaroguledo/Customer-Churn-Predictor.git
cd Customer-Churn-Predictor

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Running the analysis

```bash
jupyter lab data.ipynb           # or: jupyter notebook data.ipynb
```

Run the cells top to bottom. To execute the whole notebook headlessly:

```bash
jupyter nbconvert --to notebook --execute --inplace data.ipynb
```

## Tech Stack

| Area | Tools |
| --- | --- |
| Language | Python 3 |
| Data | pandas, NumPy |
| Modelling | scikit-learn (Logistic Regression, Decision Tree, Random Forest) |
| Preprocessing | `StandardScaler`, one-hot encoding, stratified split |
| Evaluation | `classification_report`, `roc_auc_score`, ROC / confusion-matrix displays |
| Visualisation | matplotlib, seaborn |
| Persistence | joblib |
| Testing | pytest |

## Roadmap

- [ ] Cross-validated hyperparameter tuning (`GridSearchCV` / `RandomizedSearchCV`)
- [ ] Address class imbalance with `class_weight` or SMOTE and compare
- [ ] Add gradient-boosted models (XGBoost / LightGBM)
- [ ] Wrap preprocessing and model in a single `Pipeline` / `ColumnTransformer`
- [ ] Persist the chosen model and expose a prediction script or API
- [ ] SHAP values for per-customer explanations

## License

Released under the [MIT License](https://opensource.org/licenses/MIT). Add
a `LICENSE` file to make this explicit in the repository.
