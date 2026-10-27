"""
Data ingestion, cleaning, and preprocessing module for the Elliptic Bitcoin dataset.
"""

import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

try:
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


def load_raw_data(data_dir: str = "./elliptic_bitcoin_dataset"):
    """
    Loads raw CSV files from the specified dataset directory.
    """
    edge_file = os.path.join(data_dir, "elliptic_txs_edgelist.csv")
    class_file = os.path.join(data_dir, "elliptic_txs_classes.csv")
    feat_file = os.path.join(data_dir, "elliptic_txs_features.csv")

    if not all(os.path.exists(f) for f in [edge_file, class_file, feat_file]):
        raise FileNotFoundError(
            f"Dataset files not found in {data_dir}. "
            "Please ensure elliptic_txs_edgelist.csv, elliptic_txs_classes.csv, and elliptic_txs_features.csv exist."
        )

    edges = pd.read_csv(edge_file)
    classes = pd.read_csv(class_file)
    features = pd.read_csv(feat_file, header=None)

    col_names = (
        ["txId", "time_step"]
        + [f"local_{i:02d}" for i in range(1, 94)]
        + [f"aggr_{i:02d}" for i in range(1, 73)]
    )
    features.columns = col_names

    # Map classes: 0 = Licit, 1 = Illicit, -1 = Unknown
    classes["label"] = classes["class"].replace({"unknown": -1, "2": 0, "1": 1}).astype(int)

    return edges, classes, features


def prepare_graph_data(edges, classes, features, min_timestep=0, max_timestep=20):
    """
    Filters classified transactions within the specified timesteps,
    removes isolated nodes, reindexes IDs, and returns graph tensors.
    """
    df = features.merge(classes, on="txId", how="left").sort_values("txId").reset_index(drop=True)

    # Filter classified nodes within selected timesteps
    mask = (df["label"] != -1) & (df["time_step"].between(min_timestep, max_timestep))
    valid_nodes = set(df[mask]["txId"].values)

    sub_edges = edges[
        (edges["txId1"].isin(valid_nodes)) & (edges["txId2"].isin(valid_nodes))
    ].copy()

    connected_nodes = set(sub_edges["txId1"]).union(set(sub_edges["txId2"]))
    sub_df = df[df["txId"].isin(connected_nodes)].sort_values("time_step").reset_index(drop=True)

    # Continuous reindexing
    id_map = {old: new for new, old in enumerate(sub_df["txId"].values)}
    sub_edges["txId1"] = sub_edges["txId1"].map(id_map)
    sub_edges["txId2"] = sub_edges["txId2"].map(id_map)
    sub_df["node_idx"] = sub_df["txId"].map(id_map)

    edge_index = torch.tensor(sub_edges[["txId1", "txId2"]].values.T, dtype=torch.long)
    feature_cols = [c for c in sub_df.columns if c.startswith("local_") or c.startswith("aggr_")]
    x = torch.tensor(sub_df[feature_cols].values, dtype=torch.float)
    y = torch.tensor(sub_df["label"].values, dtype=torch.long)

    if HAS_PYG:
        data = Data(x=x, edge_index=edge_index, y=y)
    else:
        data = {"x": x, "edge_index": edge_index, "y": y, "num_nodes": x.size(0), "num_features": x.size(1)}

    return data, sub_df, sub_edges


def get_stratified_split(y_labels, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Performs a stratified split on node indices ensuring identical class balance across splits.
    """
    indices = np.arange(len(y_labels))
    train_idx, test_idx, train_y, test_y = train_test_split(
        indices, y_labels, test_size=(val_ratio + test_ratio), stratify=y_labels, random_state=seed
    )
    val_sub_ratio = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx, _, _ = train_test_split(
        test_idx, test_y, train_size=val_sub_ratio, stratify=test_y, random_state=seed
    )

    return train_idx, val_idx, test_idx
