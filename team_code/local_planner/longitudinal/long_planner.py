import carla
import numpy as np

from typing import Dict, List, Tuple

from actor_prediction.collision_checker import CollisionInterval

from .config_specs import *
from .planner_algo import PlannerAlgo
from .st_occupancy import STOccupancyGrid

class LongPlanner:
    def __init__(
        self,
        st_grid_spec : STGridSpec,
        st_algo_spec : STAlgoSpec,
        algo_name : str = 'dijkstra',
        sim_freq : float = 20.0
    ):
        self.st_grid_spec = st_grid_spec
        self.st_algo_spec = st_algo_spec

        self.dt_sim = 1 / sim_freq

        self.grid_mapper = STOccupancyGrid(st_grid_spec=st_grid_spec)
        self.planner = PlannerAlgo(algo_name=algo_name, st_algo_spec=st_algo_spec)

    def run_step(
        self,
        ego_route_points_3d : np.ndarray,
        ego_speed : float,
        ego_max_speed : float,
        actor_collisions : Dict[int, List[CollisionInterval]], # K=actor id, V=collision intervals
    ) -> np.ndarray:
        # Build occupancy map
        occupancy_map = self.grid_mapper.build_st_occupancy(
            route_points_3d=ego_route_points_3d,
            max_speed=ego_max_speed,
            actor_collisions=actor_collisions,
        )

        # Find feasible path
        start_idx = 0
        goal_idx = self.grid_mapper.S_len - 1
        path = self.planner.run(occupancy_map, start_idx, goal_idx, v0=ego_speed, v_max=ego_max_speed)

        # Return empty profile if no path found
        if len(path) <= 0:
            return np.array([])

        # Convert to velocity profile
        profiles = self.extract_profiles_from_path(path, self.grid_mapper.T_arr, self.grid_mapper.S_arr, self.st_algo_spec.dt_algo_res)
        time_prof = profiles['t']
        vel_prof = profiles['v']

        # Interpolate velocity profile to match simulator
        t0 = time_prof[0]
        t1 = time_prof[-1]
        t_sim = np.arange(t0, t1 + 1e-9, self.dt_sim)

        vel_interp = np.interp(t_sim, time_prof, vel_prof)

        return vel_interp

    def extract_profiles_from_path(
        self,
        path: List[Tuple[int, int, float, float]],
        t: np.ndarray,
        s: np.ndarray,
        dt_algo_res: float
    ) -> Dict[str, np.ndarray]:
        """
        Convert a Dijkstra path [(t_idx, s_idx, v, a), ...] into time-aligned profiles.

        Returns dict with:
        t: (N,) time
        s: (N,) arclength (m)
        v: (N,) velocity (m/s)
        a: (N,) acceleration (m/s^2)
        j: (N,) jerk (m/s^3), j[0]=0 by convention
        """
        if not path:
            return {"t": np.array([]), "s": np.array([]), "v": np.array([]),
                    "a": np.array([]), "j": np.array([])}

        t_idx = np.array([p[0] for p in path], dtype=int)
        s_idx = np.array([p[1] for p in path], dtype=int)
        v     = np.array([p[2] for p in path], dtype=float)
        a     = np.array([p[3] for p in path], dtype=float)

        tt = t[t_idx]
        ss = s[s_idx]

        j = np.zeros_like(a)
        if a.size > 1:
            j[1:] = (a[1:] - a[:-1]) / float(dt_algo_res)

        return {"t": tt, "s": ss, "v": v, "a": a, "j": j}

# TODO: Add plotting/visualization methods for debugging