from .algos import *
from .config_specs import STAlgoSpec

REGISTRY = {
    'dijkstra' : STDijkstra,
}

class PlannerAlgo:
    def __init__(
        self,
        algo_name : str,
        st_algo_spec : STAlgoSpec
    ):
        if algo_name not in REGISTRY:
            raise ValueError(f"Unknown planner '{algo_name}'. Options: {sorted(REGISTRY)}")

        self.algo = REGISTRY[algo_name](st_algo_spec)

    def run(
        self,
        occupancy_grid,
        start_idx : int,
        goal_idx : int,
        **kw
    ):
        return self.algo.run(occupancy_grid, start_idx, goal_idx, **kw)