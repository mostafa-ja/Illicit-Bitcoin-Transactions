"""
Graph Convolutional Network (GCN) architectures for node classification on Bitcoin transaction graphs.
Supports both PyTorch Geometric (PyG) and pure PyTorch fallback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class PurePyGCNLayer(nn.Module):
    """
    Pure PyTorch localized spectral graph convolution layer:
    H^(l+1) = D^(-1/2) * (A + I) * D^(-1/2) * H^(l) * W^(l)
    """
    def __init__(self, in_features, out_features, bias=True):
        super(PurePyGCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        device = x.device

        # Adjacency with self-loops
        adj = torch.zeros((num_nodes, num_nodes), device=device)
        adj[edge_index[0], edge_index[1]] = 1.0
        adj += torch.eye(num_nodes, device=device)

        # Symmetric normalization D^(-1/2) A D^(-1/2)
        deg = torch.sum(adj, dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)

        norm_adj = torch.matmul(torch.matmul(D_inv_sqrt, adj), D_inv_sqrt)

        # Linear transformation and message propagation
        h = self.linear(x)
        return torch.matmul(norm_adj, h)


class GCNClassifier(nn.Module):
    """
    3-Layer Graph Convolutional Network with Batch Normalization and Dropout.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 100, out_dim: int = 2, dropout: float = 0.415):
        super(GCNClassifier, self).__init__()
        self.dropout = dropout

        if HAS_PYG:
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, out_dim)
        else:
            self.conv1 = PurePyGCNLayer(in_dim, hidden_dim)
            self.conv2 = PurePyGCNLayer(hidden_dim, hidden_dim)
            self.conv3 = PurePyGCNLayer(hidden_dim, out_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3 (Output Head)
        x = self.conv3(x, edge_index)
        return self.softmax(x)
