from typing import Tuple

from .algos import *
from .config_specs import LatAlgoSpec

REGISTRY = {
    'astar' : AStar,
}

class PlannerAlgo:
    def __init__(
        self,
        algo_name : str,
        lat_algo_spec : LatAlgoSpec
    ):
        if algo_name not in REGISTRY:
            raise ValueError(f"Unknown planner '{algo_name}'. Options: {sorted(REGISTRY)}")

        self.algo = REGISTRY[algo_name](lat_algo_spec)

    def run(
        self,
        occupancy_map,
        cost_map,
        start_node : Tuple[int, int],
        goal_node : Tuple[int, int],
        **kw
    ):
        return self.algo.run(occupancy_map, cost_map, start_node, goal_node, **kw)