import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from config import (
    NODE_FILE,
    EDGE_FILE,
    OUTPUT_TABLES,
    OUTPUT_FIGURES,
    OUTPUT_MODELS,
    OUTPUT_LOGS,
    SEEDS,
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_graph():
    nodes = pd.read_csv(NODE_FILE)
    edges = pd.read_csv(EDGE_FILE)

    labelled_mask = (
        nodes["Label_Available"].eq(1) &
        nodes["TARGET_AtRisk"].notna()
    ).to_numpy()

    node_to_index = {
        node_id: idx for idx, node_id in enumerate(nodes["Node_ID"])
    }

    source_col = "Source_Node_ID" if "Source_Node_ID" in edges.columns else edges.columns[0]
    target_col = "Target_Node_ID" if "Target_Node_ID" in edges.columns else edges.columns[1]

    edge_pairs = []
    for _, row in edges.iterrows():
        src = row[source_col]
        dst = row[target_col]
        if src in node_to_index and dst in node_to_index:
            s = node_to_index[src]
            d = node_to_index[dst]
            edge_pairs.extend([(s, d), (d, s)])

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()

    excluded = {
        "Node_ID", "Roster_Code", "School_Code", "Class_Code",
        "Label_Available", "TARGET_AtRisk"
    }
    feature_cols = [c for c in nodes.columns if c not in excluded]
    X_raw = nodes[feature_cols].copy()

    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_raw.columns if c not in numeric_cols]

    for col in numeric_cols:
        X_raw[col] = X_raw[col].fillna(X_raw[col].median())

    for col in categorical_cols:
        mode = X_raw[col].mode(dropna=True)
        X_raw[col] = X_raw[col].fillna(mode.iloc[0] if not mode.empty else "Missing")

    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), numeric_cols),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ])

    X = preprocessor.fit_transform(X_raw).astype(np.float32)
    y = nodes["TARGET_AtRisk"].fillna(-1).astype(int).to_numpy()

    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long),
    )

    metadata = {
        "node_ids": nodes["Node_ID"].tolist(),
        "labelled_mask": labelled_mask,
        "nodes": len(nodes),
        "labelled_nodes": int(labelled_mask.sum()),
        "unlabelled_nodes": int((~labelled_mask).sum()),
        "undirected_edges": int(edge_index.shape[1] // 2),
        "encoded_features": int(X.shape[1]),
    }
    return data, metadata

def make_masks(labels, labelled_mask, seed):
    labelled_indices = np.where(labelled_mask)[0]
    labelled_y = labels[labelled_indices]

    train_val_idx, test_idx = train_test_split(
        labelled_indices,
        test_size=0.20,
        stratify=labelled_y,
        random_state=seed,
    )
    train_val_y = labels[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.1875,
        stratify=train_val_y,
        random_state=seed,
    )

    n = len(labels)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, dropout=0.35):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, hidden_channels // 2, aggr="mean")
        self.classifier = torch.nn.Linear(hidden_channels // 2, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

def metrics_from_predictions(y_true, probability, prediction):
    return {
        "AUC_ROC": roc_auc_score(y_true, probability),
        "AUC_PR": average_precision_score(y_true, probability),
        "Precision_AtRisk": precision_score(y_true, prediction, zero_division=0),
        "Recall_AtRisk": recall_score(y_true, prediction, zero_division=0),
        "F1_AtRisk": f1_score(y_true, prediction, zero_division=0),
        "Weighted_F1": f1_score(y_true, prediction, average="weighted", zero_division=0),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, prediction),
        "Accuracy": accuracy_score(y_true, prediction),
        "Brier_Score": brier_score_loss(y_true, probability),
    }

def train_seed(data, metadata, seed, device, max_epochs, patience):
    set_seed(seed)
    train_mask, val_mask, test_mask = make_masks(
        data.y.numpy(), metadata["labelled_mask"], seed
    )

    graph = data.clone()
    graph.train_mask = train_mask
    graph.val_mask = val_mask
    graph.test_mask = test_mask
    graph = graph.to(device)

    model = GraphSAGE(graph.num_node_features).to(device)

    train_labels = graph.y[graph.train_mask]
    class_counts = torch.bincount(train_labels, minlength=2).float()
    class_weights = class_counts.sum() / (2.0 * class_counts.clamp_min(1.0))
    class_weights = class_weights.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    best_state = None
    best_val_auc_pr = -np.inf
    best_epoch = 0
    wait = 0
    history = []
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(graph.x, graph.edge_index)
        loss = F.cross_entropy(
            logits[graph.train_mask],
            graph.y[graph.train_mask],
            weight=class_weights,
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(graph.x, graph.edge_index)
            probability = torch.softmax(logits, dim=1)[:, 1]
            val_true = graph.y[graph.val_mask].cpu().numpy()
            val_probability = probability[graph.val_mask].cpu().numpy()
            val_auc_pr = average_precision_score(val_true, val_probability)

        history.append({
            "Seed": seed,
            "Epoch": epoch,
            "Training_Loss": float(loss.item()),
            "Validation_AUC_PR": float(val_auc_pr),
        })

        if val_auc_pr > best_val_auc_pr + 1e-6:
            best_val_auc_pr = val_auc_pr
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            break

    training_seconds = time.perf_counter() - started

    model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(graph.x, graph.edge_index)
        probability = torch.softmax(logits, dim=1)[:, 1]
        prediction = torch.argmax(logits, dim=1)

    truth = graph.y[graph.test_mask].cpu().numpy()
    test_probability = probability[graph.test_mask].cpu().numpy()
    test_prediction = prediction[graph.test_mask].cpu().numpy()

    result = metrics_from_predictions(truth, test_probability, test_prediction)
    result.update({
        "Seed": seed,
        "Best_Epoch": best_epoch,
        "Validation_AUC_PR": best_val_auc_pr,
        "Train_Seconds": training_seconds,
        "Train_N": int(graph.train_mask.sum()),
        "Validation_N": int(graph.val_mask.sum()),
        "Test_N": int(graph.test_mask.sum()),
    })

    test_indices = torch.where(graph.test_mask)[0].cpu().numpy()
    pred_rows = []
    for idx, true_value, pred_value, prob in zip(
        test_indices, truth, test_prediction, test_probability
    ):
        pred_rows.append({
            "Seed": seed,
            "Node_ID": metadata["node_ids"][idx],
            "True_Label": int(true_value),
            "Predicted_Label": int(pred_value),
            "AtRisk_Probability": float(prob),
        })

    torch.save({
        "model_state_dict": best_state,
        "input_features": graph.num_node_features,
        "seed": seed,
        "best_epoch": best_epoch,
    }, OUTPUT_MODELS / f"graphsage_baseline_seed_{seed}.pt")

    return result, history, pred_rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=40)
    args = parser.parse_args()

    for folder in [OUTPUT_TABLES, OUTPUT_FIGURES, OUTPUT_MODELS, OUTPUT_LOGS]:
        folder.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, metadata = load_graph()

    print("=" * 70)
    print("FAIRWARN-SHS STANDARD GRAPHSAGE BASELINE")
    print("=" * 70)
    print("Device:", device)
    print("Nodes:", metadata["nodes"])
    print("Labelled nodes:", metadata["labelled_nodes"])
    print("Undirected edges:", metadata["undirected_edges"])
    print("Encoded features:", metadata["encoded_features"])

    all_metrics, all_history, all_predictions = [], [], []

    for seed in SEEDS:
        print(f"\nTraining seed {seed}...")
        result, history, pred_rows = train_seed(
            data, metadata, seed, device, args.epochs, args.patience
        )
        all_metrics.append(result)
        all_history.extend(history)
        all_predictions.extend(pred_rows)
        print(
            f"AUC-PR={result['AUC_PR']:.4f} | "
            f"AUC-ROC={result['AUC_ROC']:.4f} | "
            f"Recall={result['Recall_AtRisk']:.4f} | "
            f"F1={result['F1_AtRisk']:.4f} | "
            f"Best epoch={result['Best_Epoch']}"
        )

    metrics_df = pd.DataFrame(all_metrics)
    history_df = pd.DataFrame(all_history)
    predictions_df = pd.DataFrame(all_predictions)

    metric_cols = [
        "AUC_ROC", "AUC_PR", "Precision_AtRisk", "Recall_AtRisk",
        "F1_AtRisk", "Weighted_F1", "Balanced_Accuracy",
        "Accuracy", "Brier_Score", "Train_Seconds", "Best_Epoch"
    ]

    summary = {"Model": "Standard GraphSAGE"}
    for metric in metric_cols:
        summary[f"{metric}_Mean"] = metrics_df[metric].mean()
        summary[f"{metric}_SD"] = metrics_df[metric].std(ddof=1)

    summary_df = pd.DataFrame([summary])

    metrics_df.to_csv(OUTPUT_TABLES / "graphsage_seed_metrics.csv", index=False)
    summary_df.to_csv(OUTPUT_TABLES / "graphsage_summary_mean_sd.csv", index=False)
    history_df.to_csv(OUTPUT_TABLES / "graphsage_training_history.csv", index=False)
    predictions_df.to_csv(OUTPUT_TABLES / "graphsage_test_predictions.csv", index=False)

    with open(OUTPUT_LOGS / "graphsage_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "nodes": metadata["nodes"],
            "labelled_nodes": metadata["labelled_nodes"],
            "unlabelled_nodes": metadata["unlabelled_nodes"],
            "undirected_edges": metadata["undirected_edges"],
            "encoded_features": metadata["encoded_features"],
        }, f, indent=2)

    plt.figure(figsize=(8, 5))
    for seed, group in history_df.groupby("Seed"):
        plt.plot(group["Epoch"], group["Validation_AUC_PR"], label=f"Seed {seed}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation AUC-PR")
    plt.title("Standard GraphSAGE validation performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "graphsage_validation_auc_pr.png", dpi=300)
    plt.close()

    print("\nGRAPHSAGE SUMMARY")
    print(summary_df.round(4).to_string(index=False))

if __name__ == "__main__":
    main()
