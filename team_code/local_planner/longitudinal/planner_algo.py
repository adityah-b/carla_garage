from .algos import *
from .config_specs import STAlgoSpec

from team_code.config import GlobalConfig

REGISTRY = {
    'dijkstra' : STDijkstra,
}

class PlannerAlgo:
    def __init__(
        self,
        config : GlobalConfig,
        st_algo_spec : STAlgoSpec,
        algo_name : str,
    ):
        if algo_name not in REGISTRY:
            raise ValueError(f"Unknown planner '{algo_name}'. Options: {sorted(REGISTRY)}")

        self.algo = REGISTRY[algo_name](config, st_algo_spec)

    def run(
        self,
        occupancy_map,
        cost_map,
        start_idx : int,
        goal_idx : int,
        **kw
    ):
        return self.algo.run(occupancy_map, cost_map, start_idx, goal_idx, **kw)