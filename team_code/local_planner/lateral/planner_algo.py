from typing import Tuple

from team_code.config import GlobalConfig
from team_code.local_planner.lateral.algos import *

REGISTRY = {
    'astar' : AStar,
    'sl_dijkstra' : SLDijkstra,

}

class PlannerAlgo:
    def __init__(
        self,
        algo_name : str,
        config : GlobalConfig,
    ):
        if algo_name not in REGISTRY:
            raise ValueError(f"Unknown planner '{algo_name}'. Options: {sorted(REGISTRY)}")

        self.algo = REGISTRY[algo_name](config)

    def run(
        self,
        occupancy_map,
        cost_map,
        start_idx : int,
        goal_idx : int,
        **kw
    ):
        return self.algo.run(occupancy_map, cost_map, start_idx, goal_idx, **kw)