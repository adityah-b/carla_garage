import heapq
import numpy as np
import cv2  # <-- NEW

from math import sqrt, cos, sin, atan2, pi
from dataclasses import dataclass
from typing import Tuple, List, Dict, Set, Optional, Any

from config import GlobalConfig

class AStar:
    def __init__(
        self,
        algo_spec,
    ):
        self.algo_spec = algo_spec

        # Vehicle 3-circle footprint
        L = self.algo_spec.veh_half_len_grid * 2
        W = self.algo_spec.veh_half_width_grid * 2

        self.circle_radius = sqrt(
            (L / 3) ** 2 + (W / 2) ** 2
        )

        D = 2 * sqrt(
            self.circle_radius ** 2 - (W / 2) ** 2
        )
        self.circle_offsets = np.array([-D, 0.0, D], dtype=np.float32)

    def _circle_centers_grid(
        self,
        r: float,
        c: float,
        theta_img: float
    ) -> np.ndarray:
        """
        Compute the 3 circle centers in grid coordinates (r=row, c=col)
        for a vehicle pose (r, c, theta_img), where theta_img is the
        heading in image/grid coordinates (+x right, +y down).
        Returns array of shape (3, 2): [[r1, c1], [r2, c2], [r3, c3]].
        """
        s = self.circle_offsets  # longitudinal offsets in grid cells
        cth = np.cos(theta_img)
        sth = np.sin(theta_img)

        # Along heading: dc = cos(theta), dr = sin(theta)
        centers_r = r + s * sth
        centers_c = c + s * cth

        return np.stack([centers_r, centers_c], axis=-1)  # (3, 2)

    # ---------- NEW: polygon helpers ----------

    def footprint_clear_at_index(
        self,
        r: float,
        c: float,
        theta_img: float,
        occupancy_map: np.ndarray
    ) -> bool:
        """
        Check collision at pose (r, c, theta_img) using 3-circle footprint
        in grid cells.

        occupancy_map: uint8, where 1 = obstacle, 0 = free
        (you are currently passing in obstacle_mask).
        """
        H, W = occupancy_map.shape

        centers = self._circle_centers_grid(r, c, theta_img)
        rad = self.circle_radius
        rad2 = rad * rad

        for center_r, center_c in centers:
            # Bounding box of the circle in grid indices
            row_min = int(np.floor(center_r - rad))
            row_max = int(np.ceil(center_r + rad))
            col_min = int(np.floor(center_c - rad))
            col_max = int(np.ceil(center_c + rad))

            # Clamp to map bounds
            row_min = max(0, row_min)
            row_max = min(H - 1, row_max)
            col_min = max(0, col_min)
            col_max = min(W - 1, col_max)

            # If completely outside the map, skip this circle
            if row_min > row_max or col_min > col_max:
                continue

            for rr in range(row_min, row_max + 1):
                for cc in range(col_min, col_max + 1):
                    if occupancy_map[rr, cc] == 0:
                        # 0 here would mean "no obstacle" if you passed obstacle_mask,
                        # but you currently pass obstacle_mask where 1=obstacle, 0=free.
                        # So we only care about cells == 1.
                        continue

                    # Distance from cell center to circle center (in grid units)
                    dy = (rr + 0.5) - center_r
                    dx = (cc + 0.5) - center_c
                    if dx * dx + dy * dy <= rad2:
                        # Collision with this circle
                        return False

        return True

    def segment_is_free_grid(
        self,
        r0: int, c0: int,
        r1: int, c1: int,
        occupancy_map: np.ndarray
    ) -> bool:
        """
        Sample along the straight segment from (r0,c0) → (r1,c1) in **cells**,
        using heading aligned with the motion direction.
        """
        dr = float(r1 - r0)
        dc = float(c1 - c0)
        dist_cells = float(np.hypot(dr, dc))

        if dist_cells == 0.0:
            # Arbitrary heading; 0 rad points to +x (right)
            return self.footprint_clear_at_index(r0, c0, 0.0, occupancy_map)

        theta = np.arctan2(dr, dc)  # in image coords: +y down, +x right

        if not self.footprint_clear_at_index(r1, c1, theta, occupancy_map):
            return False

        return True

    def is_goal_reached_grid(
        self,
        nr: int, nc: int,
        gr: int, gc: int,
    ) -> bool:
        if self.algo_spec.goal_tol_grid <= 0.0:
            return (nr == gr) and (nc == gc)
        return ( (nr - gr)**2 + (nc - gc)**2 ) <= (self.algo_spec.goal_tol_grid * self.algo_spec.goal_tol_grid)

    def run(
        self,
        occupancy_map : np.ndarray, # 0=occ, 1=free
        cost_map : np.ndarray,
        start_node : Tuple[int, int],
        goal_node : Tuple[int, int],
        **_: Any,
    ) -> Tuple[List[Tuple[int, int]], float]: # tuple[path(grid_x, grid_y), path_cost]
        H, W = occupancy_map.shape

        sr, sc = start_node
        gr, gc = goal_node

        def in_bounds(grid_x : int, grid_y : int) -> bool:
            return (0 <= grid_x < H) and (0 <= grid_y < W)

        def clip_to_bounds(r: int, c: int) -> Tuple[int, int]:
            return int(np.clip(r, 0, H - 1)), int(np.clip(c, 0, W - 1))

        # Heuristic cost
        def h_octile(r, c, gr, gc):
            dr, dc = abs(gr - r), abs(gc - c)
            D, D2 = 1.0, np.sqrt(2.0)
            return D * (dr + dc) + (D2 - 2 * D) * min(dr, dc)

        # Clip start and goal nodes
        sr, sc = clip_to_bounds(sr, sc)
        gr, gc = clip_to_bounds(gr, gc)

        # if not (in_bounds(sr, sc) and in_bounds(gr, gc)):
        #     return [], np.inf

        # TODO: FLIP OCCUPANCY VALUES
        obstacle_mask = (occupancy_map <= 0).astype(np.uint8)

        tol_cells = self.algo_spec.goal_tol_grid
        w_cost = self.algo_spec.w_cost

        # Moves
        # TODO: WHEN YOU FIX THE GRID AXES, REMEMBER TO FIX THESE AS WELL
        moves = [
            (0,1,1.0),
            (-1,1,np.sqrt(2)),
            (-1,0,1.0),
            # (1,0,1.0),
            (-1,-1,np.sqrt(2)),
            (0,-1,1.0),
            # (1,-1,np.sqrt(2)),
            # (1,1,np.sqrt(2))
        ]

        cmax = float(cost_map.max()) if cost_map.size else 1.0
        if cmax <= 0: cmax = 1.0
        cost_norm = cost_map.astype(np.float32) / cmax

        N = H * W
        to_idx = lambda r, c: r * W + c
        to_rc  = lambda i: (i // W, i % W)

        g = np.full(N, np.inf, dtype=np.float32)
        parent = np.full(N, -1, dtype=np.int32)
        s_idx = to_idx(sr, sc)
        g[s_idx] = 0.0

        open_heap = []
        heapq.heappush(open_heap, (h_octile(sr, sc, gr, gc), s_idx))
        closed = np.zeros(N, dtype=bool)

        while open_heap:
            f_curr, i = heapq.heappop(open_heap)
            if closed[i]:
                continue
            closed[i] = True

            r, c = to_rc(i)

            # Classic completion if tol==0 and popped the exact goal
            # if (r, c) == (gr, gc):
            if self.is_goal_reached_grid(r, c, gr, gc):
                path = []
                cur = i
                while cur != -1:
                    pr, pc = to_rc(cur)
                    path.append((pr, pc))
                    cur = parent[cur]
                path.reverse()
                return path, float(g[i])

            g_i = g[i]

            for dr, dc, dist in moves:
                nr, nc = r + dr, c + dc
                if not in_bounds(nr, nc):
                    continue

                # Quick reject if target cell is occupied
                if obstacle_mask[nr, nc] == 1:
                    continue

                # Check full-ego footprint along motion in **grid** units
                if not self.segment_is_free_grid(r, c, nr, nc, obstacle_mask):
                    continue

                # Early accept if we reached goal region and endpoint footprint is clear
                if self.is_goal_reached_grid(nr, nc, gr, gc):
                    theta = np.arctan2(dr, dc)
                    if self.footprint_clear_at_index(nr, nc, theta, obstacle_mask):
                        j = to_idx(nr, nc)
                        parent[j] = i
                        g[j] = g_i + dist * (1.0 + w_cost * cost_norm[nr, nc])

                        path = []
                        cur = j
                        while cur != -1:
                            pr, pc = to_rc(cur)
                            if cur == j:
                                path.append((gr, gc))
                            else:
                                path.append((pr, pc))
                            cur = parent[cur]
                        path.reverse()
                        return path, float(g[j])
                    # else keep searching

                # Normal relaxation
                j = to_idx(nr, nc)
                step_cost = dist * (1.0 + w_cost * cost_norm[nr, nc])
                tentative_g = g_i + step_cost

                if tentative_g < g[j]:
                    g[j] = tentative_g
                    parent[j] = i
                    f = tentative_g + h_octile(nr, nc, gr, gc)
                    heapq.heappush(open_heap, (f, j))

        return [], np.inf


    def plan_path(
        self,
        static_occupancy_map : np.ndarray,
        static_cost_map : np.ndarray,
        start_point_ego : np.ndarray,
        goal_point_ego : np.ndarray,
    ):
        start_i, start_j = self.grid.world_to_grid(start_point_ego[0], start_point_ego[1])
        goal_i, goal_j = self.grid.world_to_grid(goal_point_ego[0], goal_point_ego[1])

        path_grid, path_cost = self._run_astar(
            static_occupancy_map,
            static_cost_map,
            (start_i, start_j),
            (goal_i, goal_j)
        )
        return path_grid
