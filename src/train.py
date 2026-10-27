"""
Unified CLI execution pipeline for training and evaluating LPA and GCN models on Bitcoin transactions.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data_loader import load_raw_data, prepare_graph_data, get_stratified_split
from models.lpa import LabelPropagation
from models.gcn import GCNClassifier
from evaluate import compute_metrics, print_metrics, plot_confusion_matrices


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate AML models on Elliptic Bitcoin Dataset.")
    parser.add_argument("--data_dir", type=str, default="./elliptic_bitcoin_dataset", help="Path to raw dataset directory")
    parser.add_argument("--model", type=str, default="all", choices=["lpa", "gcn", "all"], help="Model to train/evaluate")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs for GCN")
    parser.add_argument("--hidden_dim", type=int, default=100, help="Hidden channel size for GCN")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.415, help="Dropout probability")
    parser.add_argument("--threshold", type=float, default=0.45, help="Classification decision threshold for illicit class")
    parser.add_argument("--min_timestep", type=int, default=0, help="Start timestep index")
    parser.add_argument("--max_timestep", type=int, default=20, help="End timestep index")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Executing on device: {device}")

    # 1. Load Data
    print(f"Loading dataset from: {args.data_dir}...")
    edges, classes, features = load_raw_data(args.data_dir)

    print(f"Preparing subgraph (Timesteps {args.min_timestep} to {args.max_timestep})...")
    data, sub_df, sub_edges = prepare_graph_data(edges, classes, features, args.min_timestep, args.max_timestep)

    y_numpy = sub_df["label"].values
    train_idx, val_idx, test_idx = get_stratified_split(y_numpy, 0.70, 0.15, 0.15)
    print(f"Partition counts: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    test_y = y_numpy[test_idx]
    cms = {}

    # 2. Train/Evaluate LPA
    if args.model in ["lpa", "all"]:
        print("\n>>> Evaluating Label Propagation Algorithm (LPA)...")
        edge_index = data.edge_index if hasattr(data, "edge_index") else data["edge_index"]
        y_tensor = data.y if hasattr(data, "y") else data["y"]
        num_nodes = data.num_nodes if hasattr(data, "num_nodes") else data["num_nodes"]

        lpa = LabelPropagation(edge_index, num_nodes, y_tensor, to_undirected=True)

        best_k, best_f1 = 1, 0.0
        val_y = y_numpy[val_idx]
        for k in range(1, 9):
            preds_val = lpa.predict(k, train_idx, device=args.device)
            m_val = compute_metrics(val_y, preds_val[val_idx])
            if m_val["f1"] > best_f1:
                best_f1, best_k = m_val["f1"], k

        print(f"Optimal LPA diffusion steps: k={best_k} (Val F1: {best_f1:.4f})")
        preds_all = lpa.predict(best_k, train_idx, device=args.device)
        lpa_metrics = compute_metrics(test_y, preds_all[test_idx])
        print_metrics(lpa_metrics, f"Label Propagation (k={best_k})")
        cms["LPA"] = lpa_metrics["confusion_matrix"]

    # 3. Train/Evaluate GCN
    if args.model in ["gcn", "all"]:
        print("\n>>> Training Graph Convolutional Network (GCN)...")
        x_tensor = (data.x if hasattr(data, "x") else data["x"]).to(device)
        edge_index = (data.edge_index if hasattr(data, "edge_index") else data["edge_index"]).to(device)
        y_tensor = (data.y if hasattr(data, "y") else data["y"]).to(device)

        model = GCNClassifier(
            in_dim=x_tensor.size(1),
            hidden_dim=args.hidden_dim,
            out_dim=2,
            dropout=args.dropout
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = nn.BCELoss()

        train_idx_t = torch.tensor(train_idx, device=device)
        target_one_hot = F.one_hot(y_tensor, num_classes=2).float()

        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()
            out = model(x_tensor, edge_index)
            loss = criterion(out[train_idx_t], target_one_hot[train_idx_t])
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0 or epoch == args.epochs:
                print(f"Epoch {epoch:03d}/{args.epochs} | Train Loss: {loss.item():.4f}")

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_idx_t = torch.tensor(test_idx, device=device)
            probs = model(x_tensor, edge_index)[test_idx_t, 1].cpu().numpy()
            y_pred_gcn = (probs > args.threshold).astype(int)

        gcn_metrics = compute_metrics(test_y, y_pred_gcn, probs)
        print_metrics(gcn_metrics, "Graph Convolutional Network (GCN)")
        cms["GCN"] = gcn_metrics["confusion_matrix"]

    print("Completed pipeline successfully.")


if __name__ == "__main__":
    main()
