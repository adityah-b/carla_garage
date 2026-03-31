import heapq
import numpy as np
from typing import List, Tuple, Dict, Optional, Any

from team_code.config import GlobalConfig
from team_code.local_planner.longitudinal.config_specs import STAlgoSpec

class STDijkstra:
    def __init__(
        self,
        config : GlobalConfig,
        st_algo_spec : STAlgoSpec,
    ):
       self.config = config
       self.st_algo_spec = st_algo_spec

       self.DT_SIM = 1.0 / self.config.fps

    def find_neighbours(
        self,
        occupancy_map: np.ndarray,
        cost_map: np.ndarray,
        s_idx_algo: int,   # planner s-index (coarse lattice, ds_algo)
        t_idx_grid: int,   # occupancy time index (rows)
        v_cur: float,
        a_cur: float,
        v_max: float,
        a_min: float,
        a_max: float,
        j_min: float,
        j_max: float
    ) -> List[Tuple[int, int, float, float]]:
        """
        Generate successors on a coarse s–T lattice, with collision checks on a finer s_grid × t_grid occupancy.

        - s_idx_algo lives on the planner's ds_algo lattice.
        - t_idx_grid indexes rows of occupancy_grid (dt_grid spacing).
        - occupancy_map: bool-like grid; non-zero cells are hard-blocked.
        - cost_map: float grid; non-zero cells add soft cost to the path.
        """
        neighbours: List[Tuple[int, int, float, float]] = []

        occ = occupancy_map.astype(bool)
        T_grid, S_grid = occ.shape

        # Resolutions
        ds_grid = self.st_algo_spec.ds_grid   # [m] per occupancy column
        dt_grid = self.st_algo_spec.dt_grid   # [s] per occupancy row

        ds_algo = self.st_algo_spec.ds_algo   # [m] per planner s-index
        dt_algo = self.st_algo_spec.dt_algo   # [s] per planner time step

        # Map planner dt_algo to a stride on the occupancy grid
        time_stride = max(1, int(dt_algo / dt_grid))
        next_t_idx_grid = t_idx_grid + time_stride

        if next_t_idx_grid >= T_grid:
            return neighbours

        # World extent and max index allowed on planner lattice
        S_world_max = (S_grid - 1) * ds_grid
        S_algo_max_idx = int(np.floor(S_world_max / ds_algo))
        s0_world = s_idx_algo * ds_algo

        # --- MIN/MAX ACCELERATIONS BASED ON THROTTLE AND BRAKE ---

        # Brake
        speed_kph = v_cur * 3.6
        brake_features = speed_kph ** np.arange(1, 8)
        next_speed_kph_brake = brake_features @ self.config.brake_values
        next_speed_brake = next_speed_kph_brake / 3.6
        a_min = (next_speed_brake - v_cur) / self.DT_SIM

        # Throttle
        throttle = 1.0
        throttle_features = np.array([
            speed_kph,
            speed_kph ** 2,
            throttle,
            throttle ** 2,
            speed_kph * throttle,
            speed_kph * throttle ** 2,
            speed_kph ** 2 * throttle,
            speed_kph ** 2 * throttle ** 2
        ]).T
        next_speed_kph_throttle = throttle_features @ self.config.throttle_values
        next_speed_throttle = next_speed_kph_throttle / 3.6
        a_max = (next_speed_throttle - v_cur) / self.DT_SIM

        # --- KINEMATIC BOUNDS (dI ON PLANNER LATTICE) ---

        # Velocity bounds
        s_max_vel = ((v_cur + v_max) / 2.0) * dt_algo

        di_max_vel = int(np.floor(s_max_vel / ds_algo + 1e-12))

        # Acceleration bounds
        s_min_accel = (v_cur * dt_algo) + 0.5 * (a_min * dt_algo ** 2)
        s_max_accel = (v_cur * dt_algo) + 0.5 * (a_max * dt_algo ** 2)

        di_min_acc = int(np.ceil(s_min_accel / ds_algo - 1e-12))
        di_max_acc = int(np.floor(s_max_accel / ds_algo + 1e-12))

        # Jerk bounds
        s_min_jerk = (v_cur * dt_algo) + 0.5 * (a_cur * dt_algo ** 2) + (1.0 / 6.0) * (j_min * dt_algo ** 3)
        s_max_jerk = (v_cur * dt_algo) + 0.5 * (a_cur * dt_algo ** 2) + (1.0 / 6.0) * (j_max * dt_algo ** 3)

        di_min_jerk = int(np.ceil(s_min_jerk / ds_algo - 1e-12))
        di_max_jerk = int(np.floor(s_max_jerk / ds_algo + 1e-12))

        # a_lo = a_cur - j_max * dt_algo
        # a_hi = a_cur + j_max * dt_algo
        # di_min_jerk = int(np.ceil(((v_cur * dt_algo) + (0.5 * a_lo * dt_algo * dt_algo)) / ds_algo - 1e-12))
        # di_max_jerk = int(np.floor(((v_cur * dt_algo) + (0.5 * a_hi * dt_algo * dt_algo)) / ds_algo + 1e-12))

        # Intersect ranges + monotonicity + world bound
        di_min = max(0, di_min_acc, di_min_jerk)
        di_max = min(di_max_vel, di_max_acc, di_max_jerk, S_algo_max_idx - s_idx_algo)

        if di_min > di_max:
            return neighbours

        # Candidate jumps on the planner lattice
        di_vec = np.arange(di_min, di_max + 1, dtype=np.int32)   # (N,)
        next_s_algo_vec = s_idx_algo + di_vec                    # (N,)
        v_avg_vec = (di_vec.astype(np.float32) * ds_algo) / dt_algo
        next_v_vec = 2.0 * v_avg_vec - v_cur

        # Filter out negative velocities (reversing not allowed in this lattice)
        valid_v = next_v_vec >= 0.0
        if not np.any(valid_v):
            return neighbours

        # Apply filter
        di_vec = di_vec[valid_v]
        next_s_algo_vec = next_s_algo_vec[valid_v]
        next_v_vec = next_v_vec[valid_v]
        next_a_vec = (next_v_vec - v_cur) / dt_algo

        # --- COLLISION SAMPLING ---

        # Timesteps along ST grid
        dt_grid_steps = np.linspace(0, dt_algo, time_stride + 1, dtype=np.float32)[:, None] # (R, 1)

        # Constant acceleration interpolation
        # s(t) = s0 + v_cur * t + 0.5 * a * t^2
        # We broadcast (R, 1) time steps against (1, N) accelerations
        term_v = v_cur * dt_grid_steps
        term_a = 0.5 * next_a_vec[None, :] * (dt_grid_steps ** 2)
        s_diag_world = s0_world + term_v + term_a  # (R, N)

        # Sweep the entire ego volume using offsets at the grid's resolution
        ego_half_length = self.config.ego_extent_x
        # Add ds_grid to the upper bound to ensure the +2.0 mark is included
        offsets = np.arange(-ego_half_length, ego_half_length + ds_grid, ds_grid) # (K,)

        # Broadcast s_diag_world to include the footprint offsets -> shape (R, N, K)
        s_footprint = s_diag_world[..., None] + offsets

        # Convert to occupancy columns
        i_footprint = np.floor(s_footprint / ds_grid).astype(np.int32)
        i_footprint = np.clip(i_footprint, 0, S_grid - 1) # (R, N, K)

        # Occupancy rows spanned by this step
        rows = t_idx_grid + np.arange(time_stride + 1, dtype=np.int32)
        rows = np.clip(rows, 0, T_grid - 1)

        # Check occupancy for the entire footprint volume
        # rows[:, None, None] broadcasts shape (R, 1, 1) against (R, N, K)
        hit = occ[rows[:, None, None], i_footprint]  # (R, N, K)

        # Keep trajectories where NO part of the volume hits an obstacle at ANY timestep
        # axis=(0, 2) reduces across time (R) and ego volume (K), leaving shape (N,)
        keep = ~hit.any(axis=(0, 2))

        if not np.any(keep):
            return neighbours

        next_s_algo_vec = next_s_algo_vec[keep]
        next_v_vec = next_v_vec[keep]
        next_a_vec = next_a_vec[keep]

        # Return neighbours in planner-index space + kinematics
        for s_algo, v_next, a_next in zip(next_s_algo_vec, next_v_vec, next_a_vec):
            neighbours.append((int(s_algo), next_t_idx_grid, float(v_next), float(a_next)))

        return neighbours

    def run(
        self,
        occupancy_map: np.ndarray,
        cost_map: np.ndarray,
        s_start_idx: int,   # START and GOAL are given in *grid* index space
        s_goal_idx: int,
        *,
        v0: float,
        v_max: float,
        a0: float = 0.0,
        v_cruise_ref : Optional[float] = None,
        v_goal_ref : Optional[float] = None,
        **_: Any,
    ) -> Tuple[List[Tuple[int, int, float, float]], float]:
        """
        Dijkstra on a coarse s–T lattice (ds_algo, dt_algo) with collision checks
        on a finer occupancy grid (ds_grid, dt_grid).

        occupancy_map: bool-like (T, S); non-zero cells are hard-blocked.
        cost_map: float (T, S); non-zero cells contribute soft cost.

        Returns
        -------
        path : List[(t_idx_grid, s_idx_grid, v, a)]
            Path in grid-index space with kinematics.
        total_cost : float
            Total accumulated cost of this path. If no path is found,
            returns ([], np.inf).
        """
        occ = occupancy_map.astype(bool)
        cost = cost_map.astype(float)
        K_grid, S_grid = occ.shape

        ds_grid = self.st_algo_spec.ds_grid
        dt_grid = self.st_algo_spec.dt_grid
        ds_algo = self.st_algo_spec.ds_algo
        dt_algo = self.st_algo_spec.dt_algo

        # World coords (meters) for start/goal based on occupancy grid
        s_start_world = s_start_idx * ds_grid
        s_goal_world  = s_goal_idx  * ds_grid

        # Map start/goal from grid to planner lattice
        s_start_idx_algo = int(round(s_start_world / ds_algo))
        s_goal_idx_algo  = int(round(s_goal_world  / ds_algo))

        # Guard: start cell must be free in the occupancy grid
        if not (0 <= s_start_idx < S_grid) or bool(occ[0, s_start_idx]):
            return [], float("inf")

        A_max = self.st_algo_spec.A_max
        A_min = self.st_algo_spec.A_min
        J_max = self.st_algo_spec.J_max
        J_min = self.st_algo_spec.J_min

        w_time = self.st_algo_spec.W_time
        w_vel  = self.st_algo_spec.W_vel
        w_goal_vel = self.st_algo_spec.W_goal_vel
        w_acc  = self.st_algo_spec.W_acc
        w_jerk = self.st_algo_spec.W_jerk

        a_goal_comf = A_max

        if v_cruise_ref is None:
            v_cruise_ref = v_max

        if v_goal_ref is None:
            v_goal_ref = v_cruise_ref

        def compute_stage_speed_ref(s_world: float) -> float:
            """
            Reference speed used in running cost.

            Behavior:
            - Far from goal: track v_cruise_ref
            - Near goal:
                * if goal speed is lower, cap speed by braking-feasible profile
                * if goal speed is higher, blend upward toward terminal speed
            """
            dist_rem = max(0.0, s_goal_world - s_world)  # CHANGED: remaining distance to terminal station

            if v_goal_ref < v_cruise_ref:
                # CHANGED: braking-feasible speed profile toward the terminal speed
                # v^2 <= v_goal^2 + 2*a*d
                v_goal_profile = np.sqrt(
                    max(0.0, v_goal_ref**2 + 2.0 * a_goal_comf * dist_rem)
                )
                return min(v_cruise_ref, v_goal_profile)

            return v_goal_ref

        # Get velocity quantization factor
        v_factor = self.st_algo_spec.v_quant_factor

        # Initial discrete state (quantize v0, a0 onto ds_algo lattice)
        di_prev0 = int(round((v0 * dt_algo) / ds_algo))
        ai_prev0 = int(round((a0 * dt_algo * dt_algo) / ds_algo))

        # Quantize initial velocity for the state key to prevent aliasing
        vi0 = int(round(v0 * v_factor))

        # Priority queue entries: (cost_so_far, tie_breaker, state_key)
        # state_key = (t_idx_grid, s_idx_algo, di_prev, ai_prev)
        pq: List[Tuple[float, int, Tuple[int, int, int, int]]] = []
        tie = 0

        start_key = (0, s_start_idx_algo, di_prev0, vi0)
        heapq.heappush(pq, (0.0, tie, start_key))
        tie += 1

        dist: Dict[Tuple[int, int, int, int], float] = {start_key: 0.0}
        parent: Dict[Tuple, Any] = {start_key: None}

        accel_map: Dict[Tuple, float] = {start_key: a0}
        vel_map: Dict[Tuple, float] = {start_key: v0}

        # Main loop
        while pq:
            cost_u, _, key_u = heapq.heappop(pq)
            # Skip stale entries
            if cost_u != dist.get(key_u, np.inf):
                continue

            t_idx_grid, s_idx_algo, di_prev, vi_prev = key_u

            v_cur = vel_map[key_u]
            a_cur = accel_map[key_u]

            # Goal test in world s via planner index
            if s_idx_algo >= s_goal_idx_algo:
                # reconstruct path, mapping s_idx_algo back to s_grid indices
                path = []
                curr_k = key_u
                while curr_k is not None:
                    # Reconstruct from Parent info
                    # We need to look up the v/a we stored
                    # Since parent[start] is None, handle carefully
                    p_data = parent[curr_k]

                    # Convert algo s to grid s
                    t_c, s_c_algo, _, _ = curr_k
                    s_c_world = s_c_algo * ds_algo
                    s_c_grid = int(round(s_c_world / ds_grid))

                    # If it's the start node, use v0/a0
                    if p_data is None:
                         path.append((t_c, s_c_grid, v0, a0))
                         break

                    prev_k, v_node, a_node = p_data
                    path.append((t_c, s_c_grid, v_node, a_node))
                    curr_k = prev_k

                path.reverse()
                return path, cost_u

            # No more time rows to expand
            if t_idx_grid + 1 >= K_grid:
                continue

            # Current continuous kinematics
            nbrs = self.find_neighbours(
                occupancy_map=occ,
                cost_map=cost,
                s_idx_algo=s_idx_algo,
                t_idx_grid=t_idx_grid,
                v_cur=v_cur,
                a_cur=a_cur,
                v_max=v_max,
                a_min=A_min,
                a_max=A_max,
                j_min=J_min,
                j_max=J_max,
            )

            # Expand
            for next_s_algo, next_t_idx_grid, v_next, a_next in nbrs:
                # 1. Calculate Costs
                # jerk = change in acceleration / dt
                j_curr = (a_next - a_cur) / dt_algo

                # Compute next state's world position
                s_next_world = next_s_algo * ds_algo

                # Check collision cost from grid (should be 0 or high)
                # Map next_s_algo to grid
                next_s_grid_local = int(np.floor(s_next_world / ds_grid))
                next_s_grid_local = min(max(next_s_grid_local, 0), S_grid - 1)

                obs_cost = cost[next_t_idx_grid, next_s_grid_local]

                # Get reference velocity
                v_ref_stage = compute_stage_speed_ref(s_next_world)

                cur_time = t_idx_grid * dt_grid
                next_time = next_t_idx_grid * dt_grid

                step_cost = (
                    w_time * next_time +
                    w_vel * (v_next - v_ref_stage) ** 2 +
                    # w_acc * (a_next - A_max) ** 2 +
                    # w_jerk * (j_curr ** 2) +
                    obs_cost
                )

                if s_next_world >= s_goal_world:
                    step_cost += w_goal_vel * (v_next - v_goal_ref) ** 2

                new_cost = cost_u + step_cost

                # 2. Form New Key
                # We represent the new velocity by its integer jump 'di'
                di_next = next_s_algo - s_idx_algo
                vi_next = int(round(v_next * v_factor))
                key_v = (next_t_idx_grid, next_s_algo, di_next, vi_next)

                if new_cost < dist.get(key_v, np.inf):
                    dist[key_v] = new_cost
                    # Store (Parent Key, Actual Float V, Actual Float A)
                    parent[key_v] = (key_u, v_next, a_next)
                    accel_map[key_v] = a_next
                    vel_map[key_v] = v_next

                    heapq.heappush(pq, (new_cost, tie, key_v))
                    tie += 1

        return [], float("inf")
