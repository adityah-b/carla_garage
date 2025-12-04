import heapq
import numpy as np
from typing import List, Tuple, Dict, Optional, Any

class STDijkstra:
    def __init__(
        self,
        st_algo_spec
    ):
       self.st_algo_spec = st_algo_spec

    def find_neighbours(
        self,
        occupancy_grid: np.ndarray,
        s_idx_algo: int,   # planner s-index (coarse lattice, ds_algo)
        t_idx_grid: int,   # occupancy time index (rows)
        v_cur: float,
        a_cur: float,
        ds_algo: float,
        dt_algo: float,
        v_max: float,
        a_max: float,
        j_max: float
    ) -> List[Tuple[int, int, float, float]]:
        """
        Generate successors on a coarse s–T lattice, with collision checks on a finer s_grid × t_grid occupancy.

        - s_idx_algo lives on the planner's ds_algo lattice.
        - t_idx_grid indexes rows of occupancy_grid (dt_grid spacing).
        - We sample the continuous segment in (s,t) and project to grid using ds_grid, dt_grid.
        """
        neighbours: List[Tuple[int, int, float, float]] = []

        occ = occupancy_grid
        T_grid, S_grid = occ.shape

        # Resolutions
        ds_grid = self.st_algo_spec.ds_grid   # [m] per occupancy column
        dt_grid = self.st_algo_spec.dt_grid   # [s] per occupancy row

        ds = float(ds_algo)                   # [m] per planner s-index
        dt = float(dt_algo)                   # [s] per planner time step

        # Map planner dt to a stride on the occupancy grid
        time_stride = int(dt / dt_grid)
        if time_stride <= 0:
            time_stride = 1

        next_t_idx_grid = t_idx_grid + time_stride
        if next_t_idx_grid >= T_grid:
            return neighbours

        # --- world extent and max planner index allowed ---
        S_world_max = (S_grid - 1) * ds_grid
        S_algo_max_idx = int(np.floor(S_world_max / ds))

        # --- kinematic bounds in terms of di (jump on ds_algo lattice) ---
        # Speed bound
        di_max_speed = int(np.floor((v_max * dt) / ds + 1e-12))

        # Acceleration bound
        di_min_acc = int(np.ceil(((v_cur * dt) - (a_max * dt * dt)) / ds - 1e-12))
        di_max_acc = int(np.floor(((v_cur * dt) + (a_max * dt * dt)) / ds + 1e-12))

        # Jerk bound
        a_lo = a_cur - j_max * dt
        a_hi = a_cur + j_max * dt
        di_min_jerk = int(np.ceil(((v_cur * dt) + (a_lo * dt * dt)) / ds - 1e-12))
        di_max_jerk = int(np.floor(((v_cur * dt) + (a_hi * dt * dt)) / ds + 1e-12))

        # Intersect ranges + monotonicity + world bound
        di_min = max(0, di_min_acc, di_min_jerk)
        di_max = min(di_max_speed, di_max_acc, di_max_jerk, S_algo_max_idx - s_idx_algo)

        if di_min > di_max:
            return neighbours

        # Candidate jumps on the planner lattice
        di_vec = np.arange(di_min, di_max + 1, dtype=np.int32)   # (N,)
        next_s_algo_vec = s_idx_algo + di_vec                    # (N,)
        next_v_vec = (di_vec.astype(np.float32) * ds) / dt       # (N,) m/s

        # --- Collision sampling along each candidate edge ---
        # Time samples: 0, dt_grid, 2*dt_grid, ..., dt (approx)
        dt_grid_steps = (np.arange(time_stride + 1, dtype=np.float32) * dt_grid)[:, None]  # (R,1)

        # World s at start of edge
        s0_world = s_idx_algo * ds                               # meters

        # World s along diagonals: s(t) = s0 + v_edge * Δt_k
        s_diag_world = s0_world + dt_grid_steps * next_v_vec[None, :]  # (R, N)

        # Convert to occupancy columns
        i_diag = np.floor(s_diag_world / ds_grid).astype(np.int32)
        i_diag = np.clip(i_diag, 0, S_grid - 1)

        # Also include the final landing positions as samples
        s_end_world = next_s_algo_vec.astype(np.float32) * ds
        i_end = np.floor(s_end_world / ds_grid).astype(np.int32)
        i_end = np.clip(i_end, 0, S_grid - 1)

        i_diag = np.vstack([i_diag, i_end])                      # (R+1, N)

        # Occupancy rows spanned by this step: t_idx_grid .. t_idx_grid + time_stride (+1 margin)
        rows = t_idx_grid + np.arange(time_stride + 2, dtype=np.int32)
        rows = np.clip(rows, 0, T_grid - 1)

        # Check occupancy along each diagonal
        hit = occ[rows[:, None], i_diag]                         # (R+1, N)
        keep = ~hit.any(axis=0)                                  # (N,)

        if not np.any(keep):
            return neighbours

        di_vec = di_vec[keep]
        next_s_algo_vec = next_s_algo_vec[keep]
        next_v_vec = next_v_vec[keep]

        next_a_vec = (next_v_vec - v_cur) / dt

        # Return neighbours in planner-index space + kinematics
        for s_algo, v_next, a_next in zip(next_s_algo_vec, next_v_vec, next_a_vec):
            neighbours.append((int(s_algo), next_t_idx_grid, float(v_next), float(a_next)))

        return neighbours

    def run(
        self,
        occupancy_grid: np.ndarray,
        s_start_idx: int,   # START and GOAL are given in *grid* index space
        s_goal_idx: int,
        *,
        v0: float,
        v_max: float,
        a0: float = 0.0,
        **_: Any,
    ) -> List[Tuple[int, int, float, float]]:
        """
        Dijkstra on a coarse s–T lattice (ds_algo, dt_algo) with collision checks
        on a finer occupancy grid (ds_grid, dt_grid).

        External API:
            - occupancy_grid[t, s_grid] : bool or cost-like
            - s_start_idx, s_goal_idx   : indices along s_grid

        Internal planner state:
            - (t_idx_grid, s_idx_algo, di_prev, ai_prev)
        We convert back to s_grid indices when returning the path.
        """
        occ = occupancy_grid.astype(bool)
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
        if not (0 <= s_start_idx < S_grid) or occ[0, s_start_idx]:
            return []

        A_max = self.st_algo_spec.A_max
        J_max = self.st_algo_spec.J_max

        w_vel  = self.st_algo_spec.W_vel
        w_acc  = self.st_algo_spec.W_acc
        w_jerk = self.st_algo_spec.W_jerk

        # --- helpers: di/ai (planner indices) ↔ v/a (continuous) ---
        def v_from_di(di: int) -> float:
            return (di * ds_algo) / dt_algo

        def a_from_ai(ai: int) -> float:
            return (ai * ds_algo) / (dt_algo * dt_algo)

        # Initial discrete state (quantize v0, a0 onto ds_algo lattice)
        di_prev0 = int(round((v0 * dt_algo) / ds_algo))
        ai_prev0 = int(round((a0 * dt_algo * dt_algo) / ds_algo))

        # Priority queue entries: (cost_so_far, tie_breaker, state_key)
        # state_key = (t_idx_grid, s_idx_algo, di_prev, ai_prev)
        pq: List[Tuple[float, int, Tuple[int, int, int, int]]] = []
        tie = 0

        start_key = (0, s_start_idx_algo, di_prev0, ai_prev0)
        heapq.heappush(pq, (0.0, tie, start_key))
        tie += 1

        dist: Dict[Tuple[int, int, int, int], float] = {start_key: 0.0}
        parent: Dict[Tuple[int, int, int, int], Optional[Tuple[int, int, int, int]]] = {start_key: None}

        # Optional caches, kept here if you want them later
        v_cache: Dict[Tuple[int, int, int, int], float] = {start_key: v_from_di(di_prev0)}
        a_cache: Dict[Tuple[int, int, int, int], float] = {start_key: a_from_ai(ai_prev0)}

        # Main loop
        while pq:
            cost_u, _, key_u = heapq.heappop(pq)
            if cost_u != dist.get(key_u, np.inf):
                continue  # stale

            t_idx_grid, s_idx_algo, di_prev, ai_prev = key_u

            # Goal test in world s via planner index
            if s_idx_algo * ds_algo >= s_goal_world:
                # reconstruct path, mapping s_idx_algo back to s_grid indices
                path: List[Tuple[int, int, float, float]] = []
                cur = key_u
                while cur is not None:
                    t_c, s_c_algo, di_c, ai_c = cur
                    s_c_world = s_c_algo * ds_algo
                    s_c_grid  = int(round(s_c_world / ds_grid))
                    s_c_grid  = max(0, min(S_grid - 1, s_c_grid))
                    path.append((t_c, s_c_grid, v_from_di(di_c), a_from_ai(ai_c)))
                    cur = parent[cur]
                path.reverse()
                return path

            # No more time rows to expand
            if t_idx_grid + 1 >= K_grid:
                continue

            # Current continuous kinematics
            v_cur = v_from_di(di_prev)
            a_cur = a_from_ai(ai_prev)

            # Build neighbours
            nbrs = self.find_neighbours(
                occupancy_grid=occ,
                s_idx_algo=s_idx_algo,
                t_idx_grid=t_idx_grid,
                v_cur=v_cur,
                a_cur=a_cur,
                ds_algo=ds_algo,
                dt_algo=dt_algo,
                v_max=v_max,
                a_max=A_max,
                j_max=J_max,
            )

            # Expand
            for next_s_algo, next_t_idx_grid, v_next, a_next in nbrs:
                next_s_idx_grid = np.floor((next_s_algo * ds_algo) / ds_grid).astype(np.int32)
                di_next = next_s_algo - s_idx_algo
                ai_next = di_next - di_prev
                j_next  = (a_next - a_cur) / dt_algo

                step_cost = (
                    w_vel * (v_next - v_max) ** 2 +
                    w_acc * (a_next ** 2) +
                    w_jerk * (j_next ** 2) +
                    255.0 * float(occ[next_t_idx_grid, next_s_idx_grid])  # zero for hard-occupied grids (already filtered)
                )

                key_v = (next_t_idx_grid, next_s_algo, di_next, ai_next)
                new_cost = cost_u + step_cost

                if new_cost < dist.get(key_v, np.inf):
                    dist[key_v] = new_cost
                    parent[key_v] = key_u
                    v_cache[key_v] = v_next
                    a_cache[key_v] = a_next
                    heapq.heappush(pq, (new_cost, tie, key_v))
                    tie += 1

        # No path found
        return []
