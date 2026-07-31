import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    brier_score_loss,
)
from sklearn.exceptions import ConvergenceWarning

from config import (
    NODE_FILE,
    OUTPUT_TABLES,
    OUTPUT_FIGURES,
    OUTPUT_MODELS,
    N_SPLITS,
    N_REPEATS,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def load_data():
    nodes = pd.read_csv(NODE_FILE)
    labelled = nodes[
        nodes["Label_Available"].eq(1) &
        nodes["TARGET_AtRisk"].notna()
    ].copy()
    labelled["TARGET_AtRisk"] = labelled["TARGET_AtRisk"].astype(int)

    excluded = [
        "Node_ID",
        "Roster_Code",
        "School_Code",
        "Class_Code",
        "Label_Available",
        "TARGET_AtRisk",
    ]

    feature_columns = [c for c in labelled.columns if c not in excluded]
    X = labelled[feature_columns].copy()
    y = labelled["TARGET_AtRisk"].copy()
    return labelled, X, y

def build_preprocessor(X):
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [c for c in X.columns if c not in numeric_columns]

    return ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_columns,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_columns,
        ),
    ])

def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "Feature-only MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=42,
        ),
    }

def main() -> None:
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    OUTPUT_MODELS.mkdir(parents=True, exist_ok=True)

    labelled, X, y = load_data()
    preprocessor = build_preprocessor(X)
    models = build_models()

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=42,
    )

    results = []

    print("=" * 65)
    print("TRAINING FAIRWARN-SHS FEATURE-ONLY BASELINES")
    print("=" * 65)
    print(f"Students used: {len(labelled)}")
    print(f"At-risk students: {int(y.sum())}")
    print(f"Not-at-risk students: {int((1-y).sum())}")
    print(f"Evaluation folds: {N_SPLITS * N_REPEATS}\n")

    for model_name, estimator in models.items():
        print(f"Running: {model_name}")

        for fold_number, (train_index, test_index) in enumerate(
            cv.split(X, y), start=1
        ):
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", estimator),
            ])

            started = time.perf_counter()
            pipeline.fit(X.iloc[train_index], y.iloc[train_index])
            training_seconds = time.perf_counter() - started

            probability = pipeline.predict_proba(X.iloc[test_index])[:, 1]
            prediction = (probability >= 0.5).astype(int)
            truth = y.iloc[test_index].to_numpy()

            results.append({
                "Model": model_name,
                "Fold": fold_number,
                "AUC_ROC": roc_auc_score(truth, probability),
                "AUC_PR": average_precision_score(truth, probability),
                "Precision_AtRisk": precision_score(
                    truth, prediction, zero_division=0
                ),
                "Recall_AtRisk": recall_score(
                    truth, prediction, zero_division=0
                ),
                "F1_AtRisk": f1_score(
                    truth, prediction, zero_division=0
                ),
                "Weighted_F1": f1_score(
                    truth, prediction, average="weighted", zero_division=0
                ),
                "Balanced_Accuracy": balanced_accuracy_score(
                    truth, prediction
                ),
                "Accuracy": accuracy_score(truth, prediction),
                "Brier_Score": brier_score_loss(truth, probability),
                "Train_Seconds": training_seconds,
            })

        # Fit one final full-data model only for later inspection/deployment.
        final_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        final_pipeline.fit(X, y)

        safe_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        joblib.dump(
            final_pipeline,
            OUTPUT_MODELS / f"{safe_name}.joblib"
        )

    fold_results = pd.DataFrame(results)

    metric_columns = [
        "AUC_ROC",
        "AUC_PR",
        "Precision_AtRisk",
        "Recall_AtRisk",
        "F1_AtRisk",
        "Weighted_F1",
        "Balanced_Accuracy",
        "Accuracy",
        "Brier_Score",
        "Train_Seconds",
    ]

    summary_rows = []
    for model_name, group in fold_results.groupby("Model", sort=False):
        row = {"Model": model_name}
        for metric in metric_columns:
            row[f"{metric}_Mean"] = group[metric].mean()
            row[f"{metric}_SD"] = group[metric].std(ddof=1)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    fold_results.to_csv(
        OUTPUT_TABLES / "baseline_fold_metrics.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_TABLES / "baseline_summary_mean_sd.csv",
        index=False,
    )

    plt.figure(figsize=(8, 5))
    values = [
        fold_results.loc[
            fold_results["Model"].eq(model_name), "AUC_PR"
        ].to_numpy()
        for model_name in models
    ]
    plt.boxplot(
        values,
        tick_labels=list(models.keys()),
        showmeans=True,
    )
    plt.ylabel("AUC-PR")
    plt.title("AUC-PR across 25 repeated test folds")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_FIGURES / "baseline_auc_pr_stability.png",
        dpi=300,
    )
    plt.close()

    display_columns = [
        "Model",
        "AUC_ROC_Mean",
        "AUC_ROC_SD",
        "AUC_PR_Mean",
        "AUC_PR_SD",
        "Recall_AtRisk_Mean",
        "Recall_AtRisk_SD",
        "F1_AtRisk_Mean",
        "F1_AtRisk_SD",
        "Balanced_Accuracy_Mean",
        "Balanced_Accuracy_SD",
    ]

    print("\nFINAL BASELINE SUMMARY")
    print(summary[display_columns].round(4).to_string(index=False))
    print("\nSaved outputs:")
    print(OUTPUT_TABLES / "baseline_fold_metrics.csv")
    print(OUTPUT_TABLES / "baseline_summary_mean_sd.csv")
    print(OUTPUT_FIGURES / "baseline_auc_pr_stability.png")

if __name__ == "__main__":
    main()
