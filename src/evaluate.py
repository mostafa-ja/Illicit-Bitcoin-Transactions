"""
Evaluation and metrics computation module for AML node classification.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


def compute_metrics(y_true, y_pred, y_probs=None):
    """
    Computes a comprehensive dictionary of binary classification metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    if y_probs is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_probs)
        except Exception:
            metrics["roc_auc"] = None

    return metrics


def print_metrics(metrics: dict, model_name: str = "Model"):
    """
    Prints formatted evaluation report.
    """
    print(f"\n{'='*20} {model_name} Test Results {'='*20}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    if "roc_auc" in metrics and metrics["roc_auc"] is not None:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"{'='*56}\n")


def plot_confusion_matrices(cm_dict: dict, save_path: str = None):
    """
    Plots side-by-side confusion matrix heatmaps.
    """
    num_models = len(cm_dict)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
    if num_models == 1:
        axes = [axes]

    cmaps = ["Blues", "Reds", "Greens"]

    for idx, (name, cm) in enumerate(cm_dict.items()):
        sns.heatmap(
            cm, annot=True, fmt="d", cmap=cmaps[idx % len(cmaps)],
            ax=axes[idx], xticklabels=["Licit (0)", "Illicit (1)"],
            yticklabels=["Licit (0)", "Illicit (1)"]
        )
        axes[idx].set_title(f"{name} Confusion Matrix")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
