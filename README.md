# ECT-Schizophrenia-ML

Machine learning code accompanying the publication:

> Semple, D.M., Suveges, S., Steele, J.D. (2026). *Electroconvulsive Treatment for Schizophrenia: a Decade of National Scottish Data.* BJPsych Bulletin.

## Overview

This repository contains a nested cross-validation machine learning pipeline to predict treatment outcome (CGI score at discharge) in schizophrenia patients receiving Electroconvulsive Therapy (ECT), using baseline clinical features.

**Pipeline:**
1. KNN imputation + MinMax scaling
2. Partial Least Squares (PLS) dimensionality reduction
3. Support Vector Regression (SVR) with linear kernel
4. Nested 10-fold cross-validation with inner grid search (PLS components, SVR regularisation)
5. SHAP (KernelExplainer) for feature importance
6. Quantile mapping to align predicted and observed distributions

## Requirements

Install pinned dependencies:
```bash
pip install -r requirements.txt
```

Or flexible (minimum versions):
```bash
pip install -r requirements_flexible.txt
```

**Dependencies:** `numpy`, `pandas`, `scikit-learn`, `scipy`, `shap`, `matplotlib`, `seaborn`

## Usage

```bash
python ml_ect_schizophrenia.py --data_path Dataset/dummy_dataset.csv
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--data_path` | `Dataset/dummy_dataset.csv` | Path to input CSV file |
| `--num_outer_folds` | `10` | Number of outer CV folds |
| `--num_inner_folds` | `10` | Number of inner CV folds |
| `--rand_state` | `42` | Random seed for reproducibility |

## Data format

Input CSV must contain a `CGIExit` column (target variable) and any number of feature columns. An `ID` column is dropped automatically. Missing values are handled by KNN imputation.

A dummy dataset (`Dataset/dummy_dataset.csv`) is provided to illustrate the expected format.

## Output

Running the script produces `ml_results.pdf` — a two-panel figure showing:
- **(A)** Predicted vs. observed CGI scores with Pearson r, Spearman ρ, and MAE statistics
- **(B)** SHAP feature importance (mean absolute SHAP values, top 15 features)

Example results from the manuscript are in `Manuscript_results/`.

## Licence

See [LICENSE](LICENSE).
