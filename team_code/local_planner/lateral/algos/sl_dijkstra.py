import heapq
import numpy as np
from typing import List, Tuple, Dict, Optional, Any

from team_code.config import GlobalConfig

class SLDijkstra:
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.sl_algo_spec = config.sl_algo_spec

    def run(
        self,
        occupancy_map: np.ndarray,
        cost_map : np.ndarray,
        s_start_idx : int,
        s_goal_idx : int,
        *,
        L_min : float,
        l_start_idx: int,
        l_ref_m : float = 0.0,
    ):
        S_len, L_len = occupancy_map.shape

        ds_grid = self.sl_algo_spec.ds_grid
        dl_grid = self.sl_algo_spec.dl_grid

        ds_algo = self.sl_algo_spec.ds_algo
        dl_algo = self.sl_algo_spec.dl_algo

        s_stride = max(1, int(self.sl_algo_spec.ds_algo / ds_grid))
        dl_grid_max_jump = int(np.ceil((self.sl_algo_spec.dL_max * self.sl_algo_spec.ds_algo) / dl_grid))

        pq = []
        tie = 0

        start_key = (s_start_idx, l_start_idx, l_start_idx)
        heapq.heappush(pq, (0.0, tie, start_key))
        tie += 1

        dist = {start_key: 0.0}
        parent = {start_key: None}

        best_goal_key = None
        best_goal_cost = float('inf')

        while pq:
            cost_u, _, key_u = heapq.heappop(pq)
            s_idx, l_idx, prev_l_idx = key_u

            if s_idx >= s_goal_idx:
                if cost_u < best_goal_cost:
                    best_goal_cost = cost_u
                    best_goal_key = key_u
                continue

            if best_goal_key and cost_u >= best_goal_cost:
                break

            if cost_u > dist.get(key_u, float('inf')):
                continue

            next_s_idx = min(s_idx + s_stride, S_len - 1)
            l_min_idx = max(0, l_idx - dl_grid_max_jump)
            l_max_idx = min(L_len - 1, l_idx + dl_grid_max_jump)

            prev_l_val = L_min + (prev_l_idx * dl_grid)
            cur_l_val = L_min + (l_idx * dl_grid)

            for next_l_idx in range(l_min_idx, l_max_idx + 1):
                next_l_val = L_min + next_l_idx * dl_grid
                next_dl_val = (next_l_val - cur_l_val) / ds_algo
                next_ddl_val = (next_l_val - 2 * cur_l_val + prev_l_val) / (ds_algo ** 2)

                is_occupied = False
                obs_cost = 0.0

                s_diff = next_s_idx - s_idx

                # Check every grid cell along the S-axis between current and next
                for step in range(1, s_diff + 1):
                    interp_s = s_idx + step
                    # Linearly interpolate the L-index
                    interp_l = int(round(l_idx + (next_l_idx - l_idx) * (step / s_diff)))

                    # Clamp to avoid out-of-bounds indexing at the edges
                    interp_l = max(0, min(L_len - 1, interp_l))

                    if occupancy_map[interp_s, interp_l]:
                        is_occupied = True
                        break  # Hard block found, abort edge check

                    # Keep track of the highest cost cell we pass through
                    obs_cost = max(obs_cost, cost_map[interp_s, interp_l])

                if is_occupied:
                    continue

                step_cost = (
                    self.sl_algo_spec.W_ref_offset * ((next_l_val - l_ref_m) ** 2) +
                    self.sl_algo_spec.W_heading * (next_dl_val ** 2) +
                    self.sl_algo_spec.W_curvature * (next_ddl_val ** 2) +
                    self.sl_algo_spec.W_obstacle * obs_cost
                )

                new_cost = cost_u + step_cost
                key_v = (next_s_idx, next_l_idx, l_idx)

                if new_cost < dist.get(key_v, float('inf')):
                    dist[key_v] = new_cost
                    parent[key_v] = key_u
                    heapq.heappush(pq, (new_cost, tie, key_v))
                    tie += 1

        if best_goal_key is None:
            return [], float("inf")

        path = []
        curr = best_goal_key
        while curr is not None:
            s_i, l_i, _ = curr

            s_val = s_i * ds_grid
            l_val = L_min + (l_i * dl_grid)

            path.append((s_val, l_val))
            curr = parent[curr]

        path.reverse()
        path = np.array(path, dtype=np.float32)
        return path, best_goal_cost