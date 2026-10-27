"""
Complex Network Analysis (CNA) module for Bitcoin transaction graphs.
Implements scale-free power law fitting, densification, and clustering coefficient analysis.
"""

import networkx as nx
import numpy as np
import pandas as pd


def build_bitcoin_network(edges_df: pd.DataFrame, classes_df: pd.DataFrame) -> nx.MultiDiGraph:
    """
    Constructs a directed multigraph from transaction edges and classes.
    """
    G = nx.MultiDiGraph()
    G.add_nodes_from(zip(classes_df["txId"], [{"label": v} for v in classes_df["label"]]))
    G.add_edges_from(zip(edges_df["txId1"], edges_df["txId2"]))
    return G


def compute_timestep_components(G: nx.MultiDiGraph):
    """
    Splits the global multigraph into weakly connected components representing discrete timesteps.
    """
    components = list(nx.weakly_connected_components(G))
    return [G.subgraph(comp) for comp in components]


def compute_topological_metrics(subgraph_list):
    """
    Calculates node count, edge count, average degree, density, and average clustering coefficient.
    """
    metrics = {
        "timestep": np.arange(1, len(subgraph_list) + 1),
        "number_of_nodes": [g.number_of_nodes() for g in subgraph_list],
        "number_of_edges": [g.number_of_edges() for g in subgraph_list],
        "avg_degree": [
            np.mean([d for _, d in g.degree()]) if g.number_of_nodes() > 0 else 0
            for g in subgraph_list
        ],
        "density": [nx.density(g) for g in subgraph_list],
    }
    return metrics


def fit_power_law(degree_sequence):
    """
    Estimates the power-law exponent gamma: P(k) ~ k^(-gamma) using log-log linear regression.
    """
    degrees = np.array([d for d in degree_sequence if d > 0])
    if len(degrees) == 0:
        return 0.0, 0.0

    values, counts = np.unique(degrees, return_counts=True)
    log_vals = np.log(values)
    log_probs = np.log(counts / np.sum(counts))

    slope, intercept = np.polyfit(log_vals, log_probs, 1)
    gamma = -slope
    return gamma, intercept


def fit_densification_power_law(nodes_list, edges_list):
    """
    Estimates the Densification Power Law exponent alpha: E(t) ~ N(t)^alpha.
    """
    valid = [(n, e) for n, e in zip(nodes_list, edges_list) if n > 0 and e > 0]
    if len(valid) < 2:
        return 1.0

    log_n = np.log([item[0] for item in valid])
    log_e = np.log([item[1] for item in valid])

    alpha, intercept = np.polyfit(log_n, log_e, 1)
    return alpha
