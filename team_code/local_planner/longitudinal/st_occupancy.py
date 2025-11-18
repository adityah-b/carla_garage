import cv2
import carla
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

from actor_prediction.collision_checker import CollisionInterval

class STOccupancyGrid:
    def __init__(
        self,
        st_grid_spec,
    ):
        self.st_grid_spec = st_grid_spec

        # Create s-T grid
        self.T_arr, self.S_arr = self.make_grids(st_grid_spec)

        # Store grid dimensions
        self.T_len = self.T_arr.size
        self.S_len = self.S_arr.size

    def make_grids(
        self,
        st_grid_spec
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform s–T grids."""
        T_arr = np.arange(0.0, st_grid_spec.T_max + 1e-9, st_grid_spec.dt, dtype=np.float32) # (T,)
        S_arr = np.arange(0.0, st_grid_spec.S_max + 1e-9, st_grid_spec.ds, dtype=np.float32) # (S,)
        return T_arr, S_arr

    def precompute_route_geometry(
        self,
        route_points_3d : np.ndarray
    ) -> Dict[str, np.ndarray]:
        assert route_points_3d.ndim == 2 and route_points_3d.shape[1] == 3, f"route_points_3d must be (N, 3), got {route_points_3d.shape}"
        route_xy = np.asarray(route_points_3d[:, :2], dtype=np.float32)

        seg_starts = route_xy[:-1]                       # A_i
        seg_vecs   = route_xy[1:] - route_xy[:-1]        # v_i = B_i - A_i
        seg_len    = np.linalg.norm(seg_vecs, axis=1)    # |v_i|
        seg_len2   = seg_len**2                          # |v_i|^2
        s_at_vtx   = np.concatenate(([0.0], np.cumsum(seg_len))).astype(np.float32)

        return {
            "route_xy": route_xy,
            "A": seg_starts,
            "v": seg_vecs,
            "L": seg_len,
            "L2": seg_len2,
            "s_at_vtx": s_at_vtx
        }

    def envelope_mask_under_diagonal(
        self,
        t: np.ndarray,
        s: np.ndarray,
        slope : float,
    ) -> np.ndarray:
        """valid[k,i] is True iff (k,i) lies under/on the diagonal from (0,0)→(T,s_max)."""
        K, S = t.size, s.size
        s_env = slope * t.astype(np.float64)                           # (K,)
        i_max = np.floor((s_env) / float(s[1]-s[0])).astype(int)
        i_max = np.clip(i_max, -1, S - 1)
        i_grid = np.arange(S)[None, :]
        valid = (i_grid <= i_max[:, None]) & (i_max[:, None] >= 0)

        return valid

    def build_st_occupancy(
        self,
        route_points_3d : np.ndarray,
        max_speed : float,
        actor_collisions : Dict[int, List[CollisionInterval]],
    ) -> np.ndarray:
        route_geometry = self.precompute_route_geometry(route_points_3d)

        # ---------------- 2) dynamic obstacles → occupancy ----------------
        bands = self.compute_actor_s_bands(
            route_geometry=route_geometry,
            actor_collisions=actor_collisions,
            window=2,
        )
        occ_dyn = self.build_occupancy_from_bands(self.T_arr, self.S_arr, bands_by_id=bands)

        # ---------------- 3) envelope → valid mask ----------------
        valid = self.envelope_mask_under_diagonal(self.T_arr, self.S_arr, slope=max_speed)
        occ_env = ~valid  # treat outside envelope as blocked

        # ---------------- 4) combine & (optionally) inflate ----------------
        occ_total = occ_dyn | occ_env  # boolean occupancy used by neighbours/Dijkstra

        return occ_total

    def obb_corners_world_xy(
        self,
        rows7: np.ndarray
    ) -> np.ndarray:
        """
        rows7: (T,7) with [x,y,z, ex,ey,ez, yaw_deg] (CARLA extents: ex,ey are half-dims).
        Returns: (T,4,2) world corners ordered [+ex,+ey], [+ex,-ey], [-ex,-ey], [-ex,+ey].
        """
        rows7 = np.asarray(rows7, dtype=np.float32)
        assert rows7.ndim == 2 and rows7.shape[1] == 7

        cx, cy = rows7[:, 0], rows7[:, 1]
        ex, ey = rows7[:, 3], rows7[:, 4]
        yaw = rows7[:, 6]

        th = np.deg2rad(yaw).astype(np.float32)
        c, s = np.cos(th), np.sin(th)
        R = np.stack([np.stack([c, -s], axis=1), np.stack([s, c], axis=1)], axis=1)  # (T,2,2)

        local = np.stack([
            np.stack([+ex, +ey], axis=1),
            np.stack([+ex, -ey], axis=1),
            np.stack([-ex, -ey], axis=1),
            np.stack([-ex, +ey], axis=1),
        ], axis=1).astype(np.float32)

        world = local @ np.transpose(R, (0, 2, 1))                        # rotate
        world += np.stack([cx, cy], axis=1)[:, None, :]                   # translate
        return world  # (T,4,2)

    def project_points_to_route_s(
        self,
        points_xy: np.ndarray,
        geom: dict,
        window: int = 2
    ) -> np.ndarray:
        """
        For each point:
        1) find nearest route vertex (argmin ||P - R_i||),
        2) test the segments in [i-window, i+window],
        3) pick the segment whose projection is closest to P,
        4) convert that projection to arclength s.
        """
        P = np.asarray(points_xy, dtype=np.float32)          # (M,2)
        route = geom["route_xy"]                             # (N,2)
        A, v, L, L2, s_at = geom["A"], geom["v"], geom["L"], geom["L2"], geom["s_at_vtx"]
        nseg = L.shape[0]
        assert nseg > 0, "Route must have ≥1 segment."

        # (1) nearest vertex per point
        d2 = np.sum((P[:, None, :] - route[None, :, :])**2, axis=2)  # (M,N)
        i_v = np.argmin(d2, axis=1)                                  # (M,)

        # (2) candidate segments around that vertex
        K = 2*int(window) + 1
        start = np.clip(i_v - window, 0, nseg - 1)[:, None]          # (M,1)
        cand  = np.clip(start + np.arange(K)[None, :], 0, nseg - 1)  # (M,K)

        A_c = A[cand]                  # (M,K,2)
        v_c = v[cand]                  # (M,K,2)
        L_c = L[cand]                  # (M,K)
        L2_c = L2[cand]                # (M,K)

        # (3) orthogonal projection onto each candidate segment
        #     u* = ((P-A)·v) / (v·v), u ∈ [0,1]
        w    = P[:, None, :] - A_c                                     # (M,K,2)
        num  = np.sum(w * v_c, axis=2)                                 # (M,K)  (w·v)
        denom = np.where(L2_c > 1e-12, L2_c, 1.0)                      # avoid /0
        u    = np.clip(num / denom, 0.0, 1.0)                          # (M,K)

        proj = A_c + u[..., None] * v_c                                # (M,K,2)
        dist2 = np.sum((P[:, None, :] - proj)**2, axis=2)              # (M,K)
        dist2 = np.where(L2_c > 1e-12, dist2, dist2 + 1e12)            # reject degenerate segs

        kbest = np.argmin(dist2, axis=1)                               # (M,)
        row   = np.arange(P.shape[0])
        u_b   = u[row, kbest]                                          # (M,)
        i_s   = cand[row, kbest]                                       # (M,)

        # (4) projection → arclength: s = s(i) + u*|v_i|
        s = s_at[i_s] + u_b * L[i_s]
        return s.astype(np.float32)

    def compute_actor_s_bands(
        self,
        route_geometry : Dict[str, np.ndarray],
        actor_collisions : Dict[int, List[CollisionInterval]],
        *,
        window: int = 2,
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        For each actor, project its four OBB corners per time to s, then
        take min/max across corners → s_min(t), s_max(t).
        """
        # TODO: Vectorizing colliding carla bounding boxes here for now. Future work needs to have the vectorization be done earlier in the pipeline
        def bbox_to_vec7(bb : carla.BoundingBox) -> np.ndarray:
            return np.array([
                bb.location.x, bb.location.y, bb.location.z,
                bb.extent.x,   bb.extent.y,   bb.extent.z,
                bb.rotation.yaw
            ], dtype=np.float32)

        out: Dict[int, Dict[str, np.ndarray]] = {}

        # Project collision intervals
        for vid, collisions in actor_collisions.items():
            # NOTE: Only using single collision interval (i.e. longest collision interval for now)
            collision_interval = collisions[0]
            start_idx = collision_interval.start_idx
            end_idx = collision_interval.end_idx

            # TODO: Vectorizing colliding carla bounding boxes here for now. Future work needs to have the vectorization be done earlier in the pipeline
            # TODO: No need to store colliding bboxes of source and target, only target should be fine
            bboxes : List[carla.BoundingBox] = collision_interval.collision_bboxes_b
            bboxes_arr = np.stack([bbox_to_vec7(bb) for bb in bboxes], axis=0)

            T = bboxes_arr.shape[0]

            corners = self.obb_corners_world_xy(bboxes_arr)  # (T,4,2)
            s_all   = self.project_points_to_route_s(corners.reshape(-1, 2), route_geometry, window=window)\
                        .reshape(T, 4)

            s_min = np.min(s_all, axis=1)
            s_max = np.max(s_all, axis=1)

            out[vid] = {
                "s_min": s_min.astype(np.float32),
                "s_max": s_max.astype(np.float32),
                "start_idx" : start_idx,
                "end_idx" : end_idx
            }

        return out

    def build_occupancy_from_bands(
        self,
        t: np.ndarray,
        s: np.ndarray,
        bands_by_id: Dict[int, Dict[str, np.ndarray]],
    ) -> np.ndarray:
        """
        Rasterize collision bands (per-actor) into a boolean occupancy grid occ[k,i].

        Inputs
        ------
        t : (K,) float32
            Global time grid for the planner. Indices 0..K-1.
        s : (S,) float32
            Global arclength grid for the planner. Indices 0..S-1.
        bands_by_id : dict
            For each actor id:
            {
                "s_min": (L,) float32,        # s_min at each collision timestep
                "s_max": (L,) float32,        # s_max at each collision timestep
                "start_idx": int,             # global time index (inclusive)
                "end_idx": int                # global time index (exclusive)
            }
            where L = end_idx - start_idx and arrays are aligned to that subrange.

        Output
        ------
        occ : (K, S) bool
            True where the (t_k, s_i) cell is occupied by any actor band.
            Only time rows k ∈ [start_idx, end_idx] of each actor are painted.

        Notes
        -----
        - This function assumes the collision arrays are already trimmed to the interval
        [start_idx, end_idx] (as your compute_actor_s_bands does).
        - If a band’s interval partially falls outside the planner’s [0, K-1] time window,
        it is clipped gracefully.
        - Vectorized `np.searchsorted` converts each (s_min[k], s_max[k]) to index spans
        on the S grid; we then paint contiguous slices per timestep (fast & cache-friendly).
        """
        t = t.reshape(-1)
        s = s.reshape(-1)
        K, S = t.size, s.size

        occ = np.zeros((K, S), dtype=bool)
        if not bands_by_id:
            return occ

        for id, band in bands_by_id.items():
            smin_local = band["s_min"]
            smax_local = band["s_max"]

            k_start = int(band["start_idx"])  # inclusive
            k_end   = int(band["end_idx"])    # exclusive

            # Length checks and early outs
            L = k_end - k_start
            if L <= 0 or smin_local.size == 0 or smax_local.size == 0:
                continue
            if smin_local.size != L or smax_local.size != L:
                raise ValueError(
                    f"id {id}: expected len(s_min)==len(s_max)==end_idx-start_idx+1 "
                    f"({L}), got {smin_local.size} and {smax_local.size}"
                )

            # Clip the interval to the planner horizon [0, K-1]
            k0 = max(0, k_start)
            k1 = min(K - 1, k_end)
            if k1 < k0:
                continue  # completely outside

            span = (k1 - k0)             # how many rows to paint

            # Convert s-intervals to column index ranges on the global S grid
            # TODO: Replace s[1] with ds
            j_lo = np.floor(smin_local / s[1])
            j_hi = np.floor(smax_local / s[1])

            j_lo = np.clip(j_lo, 0, S - 1)
            j_hi = np.clip(j_hi, 0, S - 1)

            # Paint each time row (contiguous slices are efficient)
            # Note: rows in occ are global indices k = k0..k1
            for r in range(span):
                lo = int(j_lo[r]); hi = int(j_hi[r])
                occ[k0 + r, lo:hi + 1] |= True

        return occ
