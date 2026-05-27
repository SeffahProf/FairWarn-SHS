# FairWarn-SHS

## A Fairness-Aware XGBoost Early Warning System for Academic Failure Prediction in Ghanaian Senior High Schools Using Primary Field Data

### Overview
This repository contains the full implementation of FairWarn-SHS — a machine learning research project developed as part of an MPhil thesis at the Department of Information Technology, KNUST, Ghana, under the supervision of Dr. Eric Opoku Osei (CANDO Lab).

FairWarn-SHS builds, evaluates, and audits a SHAP-explainable XGBoost model that predicts academic failure risk among Ghanaian Senior High School (SHS) students using primary field data collected from real SHS classrooms.

### Research Objectives
1. Collect and analyse primary field data on academic performance indicators, behavioural features, and sociodemographic characteristics of SHS students across selected Ghanaian institutions
2. Develop and evaluate a SHAP-explainable XGBoost model for predicting academic failure risk, benchmarked against three baseline classifiers
3. Audit the trained model for demographic fairness across gender and school-type subgroups using SHAP-based fairness metrics

### Research Questions
- RQ1: What are the key academic, behavioural, and sociodemographic features associated with academic failure risk among Ghanaian SHS students?
- RQ2: To what extent does a SHAP-explainable XGBoost model accurately predict academic failure risk compared to Logistic Regression, Random Forest, and Decision Tree baselines?
- RQ3: Does the XGBoost failure prediction model exhibit demographic bias across gender and school type (public vs. private)?

### Novelty
This study makes a triple contribution:
1. First primary field dataset collected from Ghanaian Senior High School students for ML failure prediction
2. First SHAP-explainable XGBoost failure prediction model applied to the Ghanaian SHS context
3. First demographic fairness audit (gender + school type) applied to an academic failure prediction model in Ghana

### Models Used
- **Primary Model:** XGBoost + SHAP (SHapley Additive exPlanations)
- **Baseline 1:** Logistic Regression
- **Baseline 2:** Random Forest
- **Baseline 3:** Decision Tree

### Evaluation Metrics
- AUC-ROC, F1-Score, Precision, Recall, Accuracy
- Confusion Matrix
- SHAP Summary Plot (global) and Force Plot (local)
- Demographic Parity Difference, Equalised Odds Difference, ABROCA (fairness audit)
- McNemar test (statistical significance, p < 0.05)

### Theoretical Framework
Self-Determination Theory (SDT) — Deci and Ryan (1985)
SDT constructs (autonomy, competence, relatedness) are mapped directly to model input features including attendance rate, CA scores, assignment submission rate, and self-reported study motivation.

### Dataset
Primary field data collected from Ghanaian Senior High School students (Form 2 and Form 3) across 3 schools — minimum 350 responses. Data collected under KNUST CHRPE ethics approval.

### Target Variable
Binary classification: At-Risk (aggregate score < 50%) vs. Not At-Risk (aggregate score ≥ 50%) — based on Ghana Education Service (GES) standard pass mark.

### Installation
Clone the repository and install dependencies:

git clone https://github.com/[YOUR-USERNAME]/FairWarn-SHS.git
cd FairWarn-SHS
pip install -r requirements.txt

### Repository Structure
FairWarn-SHS/
├── README.md
├── requirements.txt
├── data/
│   └── (anonymised dataset will be added after CHRPE approval)
├── notebooks/
│   └── (Jupyter notebooks for each pipeline step)
└── outputs/
    └── (SHAP plots, fairness metrics, model results)

### Ethics
This study was approved by the KNUST Committee on Human Research, Publication and Ethics (CHRPE). All data is fully anonymised. No personally identifiable information is stored in this repository.

### Supervisor
Dr. Eric Opoku Osei
CANDO Lab, Department of Information Technology
Kwame Nkrumah University of Science and Technology (KNUST), Ghana

### Institution
Department of Information Technology, KNUST, Ghana

### Target Journal
Computers and Education: Artificial Intelligence (Elsevier, Scopus/WoS)
