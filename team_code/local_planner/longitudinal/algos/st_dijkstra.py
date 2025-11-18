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
        s_idx: int,
        t_idx: int,
        v_cur: float,
        a_cur: float,
        ds_res: float,
        dt_res: float,
        v_max: float,
        a_max: float,
        j_max: float
    ) -> List[Tuple[int, int, float, float]]:
        """
        Vectorized successor generator for a layered s–T grid.
        Monotonicity (no backwards s) is enforced; waiting (Δs=0) is allowed if feasible.

        Returns a list of (next_s_idx, next_t_idx, next_v, next_a).

        Math (discrete to continuous mapping)
        -------------------------------------
        di = next_s_idx - s_idx                         (integer step in s-index, di >= 0)
        ds = di * ds_res
        dt = dt_res
        next_v = ds / dt
        next_a = (next_v - v_cur) / dt
        next_j = (next_a - a_cur) / dt

        Constraints → bounds on di
        --------------------------
        1) speed:     next_v ≤ v_max
                ⇒ di ≤ floor((v_max * dt)/ds_res)
        2) accel:     |next_a| ≤ a_max
                ⇒ di ∈ [ ceil((v_cur*dt - a_max*dt^2)/ds_res),
                        floor((v_cur*dt + a_max*dt^2)/ds_res) ]
        3) jerk:      |next_j| ≤ j_max  ⇔  |next_a - a_cur| ≤ j_max * dt
                ⇒ next_a ∈ [a_cur - j_max*dt, a_cur + j_max*dt]
                ⇒ di ∈ [ ceil((v_cur*dt + (a_cur - j_max*dt)*dt^2)/ds_res),
                        floor((v_cur*dt + (a_cur + j_max*dt)*dt^2)/ds_res) ]

        We intersect all di-ranges, clamp to grid, then mask by occupancy.

        Notes
        -----
        - If you disallow waiting, just set di_min = max(di_min, 1).
        - The occupancy grid can already include the velocity-envelope mask.
        """
        neighbours: List[Tuple[int, int, float, float]] = []

        # grid extents
        T, S = occupancy_grid.shape

        # next_t_idx = t_idx + 1
        next_t_idx = t_idx + int(dt_res / self.st_algo_spec.dt_grid)
        if next_t_idx >= T:
            return neighbours

        ds = float(ds_res)
        dt = float(dt_res)

        # --- convert constraints to integer bounds on di ---
        # Speed bound
        di_max_speed = int(np.floor((v_max * dt) / ds + 1e-12))

        # Acceleration bound → di range
        #   di = (v_cur*dt + next_a*dt^2) / ds
        di_min_acc = int(np.ceil(((v_cur * dt) - (a_max * dt * dt)) / ds - 1e-12))
        di_max_acc = int(np.floor(((v_cur * dt) + (a_max * dt * dt)) / ds + 1e-12))

        # Jerk bound → next_a ∈ [a_cur - j_max*dt, a_cur + j_max*dt] → di range
        a_lo = a_cur - j_max * dt
        a_hi = a_cur + j_max * dt
        di_min_jerk = int(np.ceil(((v_cur * dt) + (a_lo * dt * dt)) / ds - 1e-12))
        di_max_jerk = int(np.floor(((v_cur * dt) + (a_hi * dt * dt)) / ds + 1e-12))

        # Intersect all ranges, also enforce monotonicity (di >= 0) and grid bound
        di_min = max(0, di_min_acc, di_min_jerk)
        di_max = min(di_max_speed, di_max_acc, di_max_jerk, S - 1 - s_idx)

        if di_min > di_max:
            return neighbours

        # ---- candidate destinations ----
        di_vec  = np.arange(di_min, di_max + 1, dtype=np.int32)       # (N,)
        next_s  = s_idx + di_vec                                      # (N,)
        next_v  = (di_vec.astype(np.float32) * ds) / dt               # slope in m/s  (N,)

        # ---- diagonal sampling inside the slab (one per grid row) ----
        # absolute s at samples: s0 + v_edge * Δt_k, where Δt_k = k*grid_dt
        # TODO: MAKE THE CODE CLEARER HERE
        stride = int(dt_res / self.st_algo_spec.dt_grid)
        dt_steps = (np.arange(stride + 1, dtype=np.float32) * self.st_algo_spec.dt_grid)[:, None]  # (R,1)
        s0 = (s_idx * ds)
        s_diag = s0 + dt_steps * next_v[None, :]                      # (R,N) meters
        # print(f'\nMeasurements\n')
        # print(f'di_vec: {di_vec}')
        # print(f'next_s: {next_s}')
        # print(f'stride: {stride}')
        # print(f'v_edge: {v_edge}')
        # print(f'dt_steps: {dt_steps}')
        # print(f's0: {s0}')
        # print(f's_diag: {s_diag}')

        # station indices via floor, clamped
        i_diag = np.floor(s_diag / ds).astype(np.int32)               # (R,N)
        i_diag = np.clip(i_diag, 0, S - 1)

        i_diag = np.vstack([i_diag, next_s])

        # Row indices: t_idx + k for k in [0, 1, ..., stride + 1]
        rows = t_idx + np.arange(stride + 2, dtype=np.int32)          # (R,)
        rows = np.clip(rows, 0, T - 1)  # Clamp to valid time indices

        # print(f'rows: {rows}')
        # print(f'i_diag shape: {i_diag.shape}')
        # print(f'i_diag:\n{i_diag}')

        # Occupancy along the diagonal samples
        hit = occupancy_grid[rows[:, None], i_diag]                   # (R,N)
        keep = ~hit.any(axis=0)                                       # (N,)
        # print(f'hit: {hit}')
        # print(f'keep: {keep}')

        if not np.any(keep):
            return neighbours

        di_vec = di_vec[keep]
        next_s = next_s[keep]

        # next_s = next_s[keep]
        # v_edge = v_edge[keep]

        # # --- candidate next s-indices ---
        # di_vec = np.arange(di_min, di_max + 1, dtype=np.int32)
        # next_s = s_idx + di_vec

        # # --- occupancy pruning (False = free) ---
        # free_mask = ~occupancy_grid[next_t_idx, next_s]
        # if not np.any(free_mask):
        #     return neighbours

        # di_vec = di_vec[free_mask]
        # next_s = next_s[free_mask]

        # --- compute kinematics for the survivors (redundant with bounds, but safe) ---
        # next_v = (di_vec.astype(np.float32) * ds) / dt
        # next_a = (next_v - v_cur) / dt
        next_v = next_v[keep]
        next_a = (next_v - v_cur) / dt

        # assemble results
        neighbours = [(int(sj), next_t_idx, float(vj), float(aj))
                    for sj, vj, aj in zip(next_s, next_v, next_a)]
        return neighbours

    def run(
        self,
        occupancy_grid : np.ndarray,
        s_start_idx : int,
        s_goal_idx : int,
        *,
        v0: float,
        v_max : float,
        a0: float = 0.0,
        **_: Any,
    ) -> List[Tuple[int, int, float, float]]:
        """
        Dijkstra on a time-layered s–T grid with dynamics enforced by `findNeighbours`.

        State we store and key by (all integers):
            (t_idx, s_idx, di_prev, ai_prev)
        where di_prev = (s_idx - s_{idx-1}) in indices (0..),
                ai_prev = (di_prev - di_prevprev) in indices (can be negative).
        These let us derive:
                v_cur = di_prev * ds / dt,
                a_cur = ai_prev * ds / dt^2,
                j_next = ((di_next - di_prev - ai_prev) * ds) / dt^3.

        Returns:
            List[(t_idx, s_idx, v, a)] along the optimal path in chronological order,
            or [] if no solution was found.

        Notes:
        - The step cost is evaluated at the *landing* node (next state).
        - If you want soft obstacles, pass a *costmap* instead of a boolean grid and
        replace the last term accordingly (e.g., cost += costmap[next_t, next_s]).
        """
        occ = occupancy_grid.astype(bool)
        K, S = occ.shape
        dt = self.st_algo_spec.dt_algo_res
        ds = self.st_algo_spec.ds_algo_res

        a_max = self.st_algo_spec.A_max
        j_max = self.st_algo_spec.J_max

        w_vel = self.st_algo_spec.W_vel
        w_acc = self.st_algo_spec.W_acc
        w_jerk = self.st_algo_spec.W_jerk

        # --- helpers to go between discrete (di/ai) and continuous (v/a) ---
        def v_from_di(di: int) -> float:
            return (di * ds) / dt

        def a_from_ai(ai: int) -> float:
            return (ai * ds) / (dt * dt)

        # --- initial discrete state (quantize v0, a0 onto the grid) ---
        # di_prev0 ≈ round(v0*dt/ds), ai_prev0 ≈ round(a0*dt^2/ds)
        di_prev0 = int(round((v0 * dt) / ds))
        ai_prev0 = int(round((a0 * dt * dt) / ds))

        # Guard: start cell must be free
        if not (0 <= s_start_idx < S) or occ[0, s_start_idx]:
            return []

        # Priority queue entries: (cost_so_far, tie_breaker, state_key)
        # state_key = (t_idx, s_idx, di_prev, ai_prev)
        pq: List[Tuple[float, int, Tuple[int, int, int, int]]] = []
        tie = 0

        start_key = (0, s_start_idx, di_prev0, ai_prev0)
        heapq.heappush(pq, (0.0, tie, start_key))
        tie += 1

        dist: Dict[Tuple[int, int, int, int], float] = {start_key: 0.0}
        parent: Dict[Tuple[int, int, int, int], Optional[Tuple[int, int, int, int]]] = {start_key: None}

        # For reconstruction we also store continuous (v,a) at each node (computed once)
        v_cache: Dict[Tuple[int, int, int, int], float] = {start_key: v_from_di(di_prev0)}
        a_cache: Dict[Tuple[int, int, int, int], float] = {start_key: a_from_ai(ai_prev0)}

        # Main loop
        while pq:
            cost_u, _, key_u = heapq.heappop(pq)
            if cost_u != dist.get(key_u, np.inf):
                continue  # stale

            t_idx, s_idx, di_prev, ai_prev = key_u
            # Goal: any arrival with s >= s_goal_idx
            if s_idx >= s_goal_idx:
                # reconstruct path
                path: List[Tuple[int, int, float, float]] = []
                cur = key_u
                while cur is not None:
                    t_c, s_c, di_c, ai_c = cur
                    path.append((t_c, s_c, v_from_di(di_c), a_from_ai(ai_c)))
                    cur = parent[cur]
                path.reverse()
                return path

            # No more time layers
            if t_idx + 1 >= K:
                continue

            # Build neighbours using your dynamics checker
            v_cur = v_from_di(di_prev)
            a_cur = a_from_ai(ai_prev)

            nbrs = self.find_neighbours(
                occupancy_grid=occ,
                s_idx=s_idx,
                t_idx=t_idx,
                v_cur=v_cur,
                a_cur=a_cur,
                ds_res=ds,
                dt_res=dt,
                v_max=v_max,
                a_max=a_max,
                j_max=j_max,
            )
            # print(f'nbrs: {nbrs}')

            # Expand
            for next_s_idx, next_t_idx, v_next, a_next in nbrs:
                # derive discrete jumps
                di_next = next_s_idx - s_idx
                ai_next = di_next - di_prev
                # jerk at this landing state (continuous)
                j_next = (a_next - a_cur) / dt

                # step cost evaluated at the landing node (next)
                step_cost = (
                    w_vel * (v_next - v_max) ** 2 +
                    w_acc * (a_next ** 2) +
                    w_jerk * (j_next ** 2) +
                    255.0 * float(occ[next_t_idx, next_s_idx])  # zero for hard-occupied grids (already filtered)
                )

                key_v = (next_t_idx, next_s_idx, di_next, ai_next)
                new_cost = cost_u + step_cost

                if new_cost < dist.get(key_v, np.inf):
                    dist[key_v] = new_cost
                    parent[key_v] = key_u
                    v_cache[key_v] = v_next
                    a_cache[key_v] = a_next
                    heapq.heappush(pq, (new_cost, tie, key_v))
                    tie += 1

        # No path
        return []