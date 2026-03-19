from typing import Any

import numpy as np
import networkx as nx
import iplotx as ipx

from numpy import dtype, ndarray
from scipy import sparse


def _grid4x4_undirected_edges():
    """
    Returns undirected edges for the 4x4 grid as pairs of node indices (u,v), u < v.
    Layout (row-major):
        0  1  2  3
        4  5  6  7
        8  9 10 11
       12 13 14 15
    """
    edges = []
    for r in range(4):
        for c in range(4):
            u = r * 4 + c
            # right neighbor
            if c < 3:
                v = u + 1
                edges.append((u, v))
            # down neighbor
            if r < 3:
                v = u + 4
                edges.append((u, v))
    return edges


def weighted_directed_grid4x4(low=0.0, high=1.0, seed=None) -> sparse.csr_matrix:
    """
    Produces a directed weighted and unweighted adjacency for the grid:
      - For every undirected neighbor pair {u,v}, creates TWO directed edges u->v and v->u
      - Weights are in (low, high) open interval (default (0,1))
    :param low: lower bound
    :param high: upper bound
    :param seed: random seed
    :return: Two directed graphs, one with weights and one without.
    """
    rng = np.random.default_rng(seed)

    if not (low < high):
        raise ValueError("Require low < high.")

    # draw strictly inside ]low, high[
    def draw_weight():
        # rng.random() is in [0,1[; this maps to [low, high[ (practically open on floats)
        w = low + (high - low) * float(rng.random())
        # avoid exact low (vanishingly unlikely), bump if desired
        if w <= low:
            w = np.nextafter(low, high)
        return w

    undirected = _grid4x4_undirected_edges()

    weighted_graph: ndarray[tuple[int, int], dtype[np.float64]] = np.zeros((16, 16), dtype=np.float64)

    # add edges
    for u, v in undirected:
        w_uv = draw_weight()
        w_vu = draw_weight()

        weighted_graph[u, v] = w_uv
        weighted_graph[v, u] = w_vu

    # empty diagonals, just in case
    np.fill_diagonal(weighted_graph, 0.0)

    return sparse.csr_matrix(weighted_graph)


# === Example ===
if __name__ == "__main__":
    A = weighted_directed_grid4x4(seed=42)
    G = nx.from_scipy_sparse_array(A, create_using=nx.DiGraph, edge_attribute="weight")
    print(A)
    print(G)
    layout = nx.layout.spring_layout(G)
    ipx.network(G, layout=layout, show=True)
