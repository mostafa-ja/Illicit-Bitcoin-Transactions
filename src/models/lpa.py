"""
Label Propagation Algorithm (LPA) implementation for semi-supervised graph node classification.
Based on absorbing random walks and transductive diffusion (Zhu & Ghahramani, 2002).
"""

import torch
import torch.nn.functional as F
import numpy as np


class LabelPropagation:
    def __init__(self, edge_index: torch.Tensor, num_nodes: int, y_labels: torch.Tensor, to_undirected: bool = True):
        """
        Initializes LPA with graph adjacency and ground-truth labels.
        """
        if to_undirected:
            # Make graph undirected to allow bi-directional diffusion
            rev_edges = torch.stack([edge_index[1], edge_index[0]], dim=0)
            edge_index = torch.cat([edge_index, rev_edges], dim=1)
            edge_index = torch.unique(edge_index, dim=1)

        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.y = y_labels

    def predict(self, k: int, train_indices, device="cpu"):
        """
        Propagates labels for k diffusion steps.
        T = D^(-1) * (A + I)
        """
        device = torch.device(device)
        
        # Build dense adjacency matrix with self-loops
        A = torch.zeros((self.num_nodes, self.num_nodes), device=device)
        src = self.edge_index[0].to(device)
        dst = self.edge_index[1].to(device)
        A[src, dst] = 1.0
        A += torch.eye(self.num_nodes, device=device)

        # Inverted Degree Matrix D^(-1)
        deg = torch.sum(A, dim=1)
        deg_inv = torch.reciprocal(deg)
        deg_inv[deg_inv == float("inf")] = 0.0
        D_inv = torch.diag(deg_inv)

        # Transition Probability Matrix
        T = torch.matmul(D_inv, A)

        # Initial one-hot label distribution
        Y = torch.zeros((self.num_nodes, 2), device=device)
        train_y_dev = self.y[train_indices].to(device)
        Y[train_indices] = F.one_hot(train_y_dev, num_classes=2).float()

        # Iterative diffusion with ground-truth clamping
        for _ in range(k):
            Y_next = torch.matmul(T, Y)
            Y_next[train_indices] = F.one_hot(train_y_dev, num_classes=2).float()
            Y = Y_next

        predictions = torch.argmax(Y, dim=1)
        return predictions.cpu().numpy()
