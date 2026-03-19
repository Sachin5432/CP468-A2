import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csgraph

from sim.world import World


class Agent(object):
    current_pos: np.int32
    target_lot: np.int32
    destination: int
    _max_walking_dist: float
    _distance_to_target: np.float64
    _curr_path: list[np.int32]
    _street_bfo: list[np.int32]
    _dijkstra_output: Any
    _unweighted_distance: Any

    def __init__(self, start=0, dest=15, max_walking_dist=2.0):
        self.current_pos: np.int32 = np.int32(start)
        self.target_lot: np.int32 = np.int32(0)
        self.destination: int = dest
        self._max_walking_dist = max_walking_dist
        self._distance_to_target = np.float64(-9999)
        self._curr_path: list[np.int32] = []
        self._street_bfo = []
        self._dijkstra_output = None
        self._unweighted_distance = None

    def act(self, world: World) -> bool:
        """
        Makes the agent act.

        :param world: World object containing all information about the environment.
        :return: Returns true if the agent is done.
        """

        print(f"[{datetime.datetime.now().astimezone()}] Agent current position = {self.current_pos}.")

        self._compute_distances(world)
        self._find_lot(world)

        dist, predecessors = self._dijkstra_output

        # reconstruct path by walking through predecessors backwards
        path = []
        current = self.target_lot
        while current != -9999:  # scipy sentinel for no predecessor
            path.append(current)
            if current == self.current_pos:
                break
            current = predecessors[current]

        self._curr_path = path[::-1]
        print(f"[{datetime.datetime.now().astimezone()}] Agent selected path = {self._curr_path}")

        self.current_pos = self._curr_path[1]  # go to the next optimal node
        print(f"[{datetime.datetime.now().astimezone()}] Agent new position = {self.current_pos}")

        return False

    def finished(self):
        return self.current_pos == self.target_lot

    def _compute_distances(self, world: World):
        """
        Computes the distances between relevant nodes using Dijkstra's algorithm and Floyd Warshall algorithm for the
        purpose of the agent's cost evaluation.

        :param world: World object containing all information about the environment.
        """

        print(f"[{datetime.datetime.now().astimezone()}] Computing distances...")

        map_copy = world.get_map()

        # O(n)
        self._street_bfo = csgraph.breadth_first_order(map_copy, self.destination, directed=True)[0]

        # determine the shortest path and distance from current_pos to all other nodes for the car.
        # O(E log(V))
        # output = (distance: matrix, predecessors: list)
        self._dijkstra_output = csgraph.dijkstra(map_copy,
                                                 directed=True,
                                                 indices=self.current_pos,
                                                 return_predecessors=True)

        # walking ignores car traffic, thus unweighted distances are used.
        # O(V^3)
        self._unweighted_distance = csgraph.floyd_warshall(map_copy,
                                                           directed=True,
                                                           unweighted=True,
                                                           return_predecessors=False)

    def _find_lot(self, world: World):
        """
        Finds a suitable parking lot. The lot is either the nearest lot that is within the acceptable walking distance
        from the destination. Or the parking lot closest to the destination, if all parking lots are outside the
        acceptable walking distance.

        :param world: World object containing all information about the environment.
        """

        print(f"[{datetime.datetime.now().astimezone()}] Finding parking lot...")

        # find parking lots closest to the destination via walking, filtering out lots that are beyond the acceptable walking distance.
        lots_near_dest = [
            i for i in self._street_bfo
            if world.is_parking_lot(i) and (self._unweighted_distance[i, self.destination] <= self._max_walking_dist)
        ]

        if lots_near_dest:
            # get the parking lot that's within max walking distance that's closest to the agent.
            dists, _ = self._dijkstra_output

            # evaluate costs
            cost_df = pd.DataFrame([
                    (lot_pos, dists[lot_pos], world.get_cost_for_lot(lot_pos))
                    for lot_pos
                    in lots_near_dest
                ],
                columns=['node', 'distance', 'cost']
            )

            # pick the lowest cost
            cost_df.sort_values(by=["distance", "cost"], ascending=[True, True])
            min_distance = cost_df.iloc[0]["node"]
            self.target_lot = np.int32(min_distance)

            print(f"[{datetime.datetime.now().astimezone()}] Found good lot at node = {self.target_lot}, distance = {cost_df.iloc[0]["distance"]}.")
        else:
            # get the parking lot that's closest to the destination if there are no lots sufficiently close to the destination.
            self.target_lot = [iii for iii in self._street_bfo if world.is_parking_lot(iii)][0]
            print(f"[{datetime.datetime.now().astimezone()}] Found compromise lot at {self.target_lot}.")
