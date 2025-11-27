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

    # ---------- NEW: polygon helpers ----------

    def rect_polygon_grid(
        self,
        r: float,
        c: float,
        theta_img: float
    ) -> np.ndarray:
        """
        Rotated rectangle polygon in image coords (x=col, y=row) in **cells**.
        Returns float32 array of shape (4,2): [ [x,y], ... ] in order:
        front-left, front-right, rear-right, rear-left.
        """
        x_half = self.algo_spec.veh_half_len_grid
        y_half = self.algo_spec.veh_half_width_grid

        # Local rectangle in (x=cols, y=rows)
        poly_local = np.array([
            [x_half, y_half],
            [x_half, -y_half],
            [-x_half, -y_half],
            [-x_half, y_half],
        ], dtype=np.float32)

        cth, sth = np.cos(theta_img), np.sin(theta_img)
        R = np.array([[cth, -sth],
                      [sth,  cth]], dtype=np.float32)  # standard 2D rotation

        poly_rc = (poly_local @ R.T)
        # Translate to center (c, r): x=col, y=row
        poly_rc[:, 0] += c
        poly_rc[:, 1] += r
        return poly_rc  # float32, shape (4,2)

    def poly_oob(
        self,
        poly_rc: np.ndarray,
        H: int,
        W: int
    ) -> bool:
        """Return True if polygon’s bounding box goes out-of-bounds."""
        xs = poly_rc[:, 0]; ys = poly_rc[:, 1]
        minx, maxx = xs.min(), xs.max()
        miny, maxy = ys.min(), ys.max()
        return (minx < 0) or (miny < 0) or (maxx >= (W - 1)) or (maxy >= (H - 1))

    def footprint_clear_at_index(
        self,
        r: float,
        c: float,
        theta_img: float,
        occupancy_map: np.ndarray
    ) -> bool:
        """
        Rasterize ego footprint directly on the grid (cells) and check overlap.
        occupancy_map: uint8 (1=obstacle, 0=free)
        """
        H, W = occupancy_map.shape
        poly = self.rect_polygon_grid(r, c, theta_img)

        # Treat any out-of-bounds polygon as collision (conservative).
        # if self.poly_oob(poly, H, W):
        #     return False

        # cv2 wants int32 points shaped (N,1,2) with (x=col, y=row)
        pts = poly.astype(np.int32).reshape(-1, 1, 2)

        fp_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(fp_mask, [pts], 1)

        # occ_img = (occupancy_map * 255).astype(np.uint8)
        # occ_bgr = cv2.cvtColor(occ_img, cv2.COLOR_GRAY2BGR)

        # fp_img = (fp_mask * 255).astype(np.uint8)
        # fp_bgr = cv2.cvtColor(fp_img | occ_img, cv2.COLOR_GRAY2BGR)

        # maps_img = np.hstack([occ_bgr, fp_bgr])
        # cv2.namedWindow("BirdView Maps", cv2.WINDOW_NORMAL)
        # cv2.imshow('BirdView Maps', maps_img)
        # cv2.waitKey(1)

        return not np.any((fp_mask & occupancy_map) != 0)

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

        # Heuristic cost
        def h_octile(r, c, gr, gc):
            dr, dc = abs(gr - r), abs(gc - c)
            D, D2 = 1.0, np.sqrt(2.0)
            return D * (dr + dc) + (D2 - 2 * D) * min(dr, dc)

        if not (in_bounds(sr, sc) and in_bounds(gr, gc)):
            return [], np.inf

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
