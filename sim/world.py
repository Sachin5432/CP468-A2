import copy
import datetime

import iplotx as ipx
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sim import gen_matrix


class World(object):
    _street_graph: csr_matrix
    _parking_lots: pd.DataFrame

    def __init__(self, seed=None):
        """
        World object to manage environment.

        :param seed: Seed for random number generator, defaults to None.
        """
        print(f"[{datetime.datetime.now().astimezone()}] Initializing world...")

        rng: np.random.Generator = np.random.default_rng(seed)

        self._street_graph = gen_matrix.weighted_directed_grid4x4(seed=seed)
        self._parking_lots = pd.DataFrame({
            "position": rng.choice(np.arange(0, 16, dtype=np.int32), size=3),
            "cost": rng.random(size=3),
        })

        print(f"[{datetime.datetime.now().astimezone()}] Parking lots: {list(self._parking_lots["position"])}")

    def draw_street_graph(self):
        G = nx.from_scipy_sparse_array(self._street_graph, create_using=nx.DiGraph, edge_attribute="weight")
        layout = nx.spring_layout(G)
        ipx.network(G, layout=layout, show=True)

    def update(self):
        """
        Updates the environment's state with new data.
        """

        print(f"[{datetime.datetime.now().astimezone()}] Updating street graph...")
        self._street_graph = gen_matrix.weighted_directed_grid4x4(seed=None)  # update the code so a new graph is generated

    def get_map(self) -> csr_matrix:
        return copy.copy(self._street_graph)

    def is_parking_lot(self, i) -> bool:
        return i in list(self._parking_lots["position"])

    def get_parking_lots(self) -> pd.DataFrame:
        return copy.copy(self._parking_lots)

    def get_cost_for_lot(self, lot_no: np.int32) -> np.float64:
        return self._parking_lots[self._parking_lots["position"] == lot_no]["cost"].iloc[0]
