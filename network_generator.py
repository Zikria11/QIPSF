# network_generator.py
import numpy as np
import networkx as nx

def initialize_prob_matrix(n_nodes, p0, seed=None):
    rng = np.random.default_rng(seed)
    P = rng.random((n_nodes, n_nodes))
    P = (P < p0).astype(float)
    np.fill_diagonal(P, 0.0)
    P = np.triu(P, 1)
    P = P + P.T
    return P


def apply_backbone_floor(P, backbone_edges, p_min=0.1):
    """
    Enforce a minimum probability p_min on a set of backbone edges to avoid
    total disconnection.
    backbone_edges: list of (i,j) tuples.
    """
    P_new = P.copy()
    for (i, j) in backbone_edges:
        if i > j:
            i, j = j, i
        P_new[i, j] = max(P_new[i, j], p_min)
        P_new[j, i] = P_new[i, j]
    return P_new


def sample_graph_from_probabilities(P, seed=None):
    n = P.shape[0]
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < P[i, j]:
                G.add_edge(i, j)

    return G
