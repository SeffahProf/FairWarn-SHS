# FairWarn-SHS

**FairWarn-SHS: A Fairness-Aware GraphSAGE Early Warning System for Academic Failure Prediction in Ghanaian Senior High Schools Using Primary Field Data**

This repository contains the reproducible experiment pipeline for the FairWarn-SHS thesis.

## Current stage

Completed:

1. Data correction and validation
2. Node-feature preparation
3. Peer-network edge-list preparation
4. Logistic Regression baseline
5. Random Forest baseline
6. Feature-only Multi-Layer Perceptron baseline

Next:

7. Standard GraphSAGE baseline
8. Fairness-aware GraphSAGE contribution
9. SHAP/GNN explainability
10. Fairness audit
11. Ablation and statistical testing

## Repository structure

```text
FairWarn-SHS/
├── data/
│   ├── raw/
│   ├── corrected/
│   └── processed/
├── notebooks/
├── src/
├── outputs/
│   ├── tables/
│   ├── figures/
│   ├── models/
│   └── logs/
├── requirements.txt
├── run_baselines.py
└── README.md
```

## What the three baseline models mean

### Logistic Regression

This is the simplest statistical model. It estimates how each input variable changes the probability that a student is at risk.

### Random Forest

This model builds many decision trees and combines their decisions. It is useful for capturing nonlinear relationships between student characteristics and academic risk.

### Feature-only MLP

This is a neural network that uses student features but does not use peer-network connections. It is the most important direct comparator for GraphSAGE because both are neural models, but only GraphSAGE receives graph edges.

## Evaluation design

The baseline experiment uses repeated stratified five-fold cross-validation:

- 5 folds
- 5 repeats
- 25 test-fold results per model
- Mean and standard deviation reported for every metric

All imputation, scaling, and encoding operations are fitted inside the training fold only.

## Main metrics

- **AUC-ROC:** how well the model separates at-risk and not-at-risk students across thresholds.
- **AUC-PR:** how well the model detects at-risk students while controlling false alarms.
- **Recall:** the percentage of truly at-risk students detected.
- **Precision:** the percentage of flagged students who are truly at risk.
- **F1-score:** balance between precision and recall.
- **Balanced accuracy:** average recall across both classes.
- **Brier score:** probability calibration error; lower is better.

## Running locally on Windows

Open Command Prompt or PowerShell inside the repository folder.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Inspect the dataset:

```bash
python src/01_inspect_data.py
```

Train all feature-only baselines:

```bash
python src/02_train_baselines.py
```

Or run both stages together:

```bash
python run_baselines.py
```

## Running in Google Colab

1. Open Google Colab.
2. Upload `notebooks/03_baseline_models.ipynb`.
3. Upload `data/processed/FairWarn_SHS_Node_Features.csv` when requested.
4. Select **Runtime → Run all**.
5. Download the generated CSV outputs.

## Current preliminary results

The current experiment used 396 labelled students.

| Model | Mean AUC-ROC | Mean AUC-PR | Mean At-Risk Recall | Mean At-Risk F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9654 | 0.9332 | 0.8666 | 0.8379 |
| Random Forest | 0.9711 | 0.9556 | 0.8457 | 0.8762 |
| Feature-only MLP | 0.9552 | 0.9200 | 0.7583 | 0.8119 |

These values are preliminary baselines. They are not yet the final FairWarn-SHS results because GraphSAGE, fairness auditing, ablation, and strict early-warning tests remain.

## Important interpretation

A high score does not automatically prove that the full thesis contribution is successful. It only shows that the current student features contain strong predictive information. The final thesis must still determine:

- whether GraphSAGE improves on the feature-only MLP;
- whether peer relationships add useful information;
- whether predictions are fair across gender and school type;
- whether results remain strong when direct academic scores are excluded;
- whether the fairness-aware contribution provides a measurable gain or trade-off.

## Research integrity

Do not manually alter generated metrics. Every number reported in the thesis must be traceable to a saved output file produced by the code.
