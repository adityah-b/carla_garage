import carla
import numpy as np

from typing import Dict, Optional, Tuple, List

from actor_prediction.collision_checker import CollisionInterval

class STOccupancyGrid:
    def __init__(self, st_grid_spec) -> None:
        self.st_grid_spec = st_grid_spec

        # Create s-T grid
        self.T_arr, self.S_arr = self.make_grids(st_grid_spec)

        # Store grid dimensions
        self.T_len = self.T_arr.size
        self.S_len = self.S_arr.size

        # Envelope cache (reused unless slope changes)
        self._envelope_slope: Optional[float] = None
        self._envelope_mask: Optional[np.ndarray] = None

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
        ds = float(s[1] - s[0]) if S > 1 else 1.0
        i_max = np.floor((s_env) / ds).astype(int)
        i_max = np.clip(i_max, -1, S - 1)
        i_grid = np.arange(S)[None, :]
        valid = (i_grid <= i_max[:, None]) & (i_max[:, None] >= 0)

        return valid

    def get_envelope_mask(
        self,
        slope: float
    ) -> np.ndarray:
        """Return cached envelope if slope unchanged; otherwise recompute."""
        if self._envelope_mask is not None and self._envelope_slope == slope:
            return self._envelope_mask

        self._envelope_mask = self.envelope_mask_under_diagonal(self.T_arr, self.S_arr, slope=slope)
        self._envelope_slope = slope

        return self._envelope_mask

    def build_st_occupancy(
        self,
        route_points_3d : np.ndarray,
        max_speed : float,
        actor_collisions : Dict[int, List[CollisionInterval]],
        all_conditions : Dict[int, Tuple] = {}
    ) -> np.ndarray:
        route_geometry = self.precompute_route_geometry(route_points_3d)

        # ---------------- 2) dynamic obstacles → occupancy ----------------
        bands = self.compute_actor_s_bands(
            route_geometry=route_geometry,
            actor_collisions=actor_collisions,
            window=2,
            all_conditions=all_conditions
        )
        cost_dyn = self.build_costmap_from_bands(self.T_arr, self.S_arr, bands_by_id=bands)

        # ---------------- 3) envelope → valid mask ----------------
        valid = self.get_envelope_mask(max_speed)
        occ_env = ~valid  # treat outside envelope as blocked

        # ---------------- 4) combine & (optionally) inflate ----------------
        # occ_total = occ_dyn | occ_env  # boolean occupancy used by neighbours/Dijkstra
        cost_total = np.maximum(cost_dyn, occ_env.astype(np.float32))  # boolean occupancy used by neighbours/Dijkstra

        return cost_total

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
        all_conditions: Dict[int, Tuple] = {},
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        For each actor, project its four OBB corners per time to s, then
        take min/max across corners → s_min(t), s_max(t).

        `all_conditions` lets callers describe how to extend and weight the costs
        for specific actors using (action_type, actor_type, importance):
            - action_type ∈ {"yield_for", "watch_out_for"}
            - actor_type  ∈ {"vehicle", "cyclist", "ped"}
            - importance  ∈ [0.0, 1.0]

        "yield_for" emphasises waiting until the actor passes the conflict region
        by padding the time dimension primarily before the collision interval.
        "watch_out_for" builds symmetric time padding and a larger spatial buffer
        around the collision band. Actor type and importance scale the collision
        cost and dilation to reflect perceived risk.
        """
        # TODO: Vectorizing colliding carla bounding boxes here for now. Future work needs to have the vectorization be done earlier in the pipeline
        def bbox_to_vec7(bb : carla.BoundingBox) -> np.ndarray:
            return np.array([
                bb.location.x, bb.location.y, bb.location.z,
                bb.extent.x,   bb.extent.y,   bb.extent.z,
                bb.rotation.yaw
            ], dtype=np.float32)

        out: Dict[int, Dict[str, np.ndarray]] = {}

        # Number of discrete time indices to extend before/after the true
        # collision interval for each action type.
        time_extension_steps = {
            # "yield_for": (2, 10),      # wait longer after the collision
            "yield_for": (100, 10),      # wait longer after the collision
            "watch_out_for": (10, 10),   # symmetric caution window
        }

        # Spatial dilation (in metres) based on actor type and action flavour.
        base_space_dilation = 0.5
        space_dilation_by_actor_type = {"vehicle": 0.6, "cyclist": 0.9, "ped": 1.2}
        space_dilation_by_action = {"yield_for": 0.5, "watch_out_for": 0.25}

        # Collision cost scaling based on actor type + importance.
        collision_cost_scale = {"vehicle": 1.0, "cyclist": 1.2, "ped": 1.4}

        # Project collision intervals
        for actor_id, collisions in actor_collisions.items():
            # NOTE: Only using single collision interval (i.e. longest collision interval for now)
            collision_interval = collisions[0]
            start_idx = collision_interval.start_idx
            end_idx = collision_interval.end_idx

            # TODO: Vectorizing colliding carla bounding boxes here for now. Future work needs to have the vectorization be done earlier in the pipeline
            # TODO: No need to store colliding bboxes of source and target, only target should be fine
            bboxes : List[carla.BoundingBox] = collision_interval.collision_bboxes_b
            bboxes_arr = np.stack([bbox_to_vec7(bb) for bb in bboxes], axis=0)

            action_type: Optional[str] = None
            actor_type: Optional[str] = None
            importance: float = 0.0
            if actor_id in all_conditions:
                action_type, actor_type, importance = all_conditions[actor_id]

            # Determine time padding driven by the action type (if any)
            pad_before = pad_after = 0
            if action_type is not None:
                pad_before, pad_after = time_extension_steps[action_type.value]

            extended_start = max(0, start_idx - pad_before)
            extended_end = min(bboxes_arr.shape[0] - 1, end_idx + pad_after)

            extended_indices = np.arange(extended_start, extended_end, dtype=np.int32)
            arr_extended = bboxes_arr[extended_indices]
            T_ext = arr_extended.shape[0]

            corners_ext = self.obb_corners_world_xy(arr_extended)
            s_all_ext = self.project_points_to_route_s(
                corners_ext.reshape(-1, 2), route_geometry, window=window
            ).reshape(T_ext, 4)

            s_min_ext = np.min(s_all_ext, axis=1)
            s_max_ext = np.max(s_all_ext, axis=1)

            collision_mask = (extended_indices >= start_idx) & (extended_indices <= end_idx)
            time_distance_steps = np.zeros_like(extended_indices, dtype=np.float32)
            if collision_mask.any():
                # Distance in discrete steps to the nearest point inside the collision interval
                before_dist = np.maximum(0, start_idx - extended_indices)
                after_dist = np.maximum(0, extended_indices - end_idx)
                time_distance_steps = np.maximum(before_dist, after_dist).astype(np.float32)

            action_dilation = space_dilation_by_action.get(action_type, 0.0)
            actor_dilation = space_dilation_by_actor_type.get(actor_type, 0.0)
            space_dilation = base_space_dilation + action_dilation + actor_dilation
            cost_scale = collision_cost_scale.get(actor_type, 1.0) * (1.0 + 0.75 * importance)

            out[actor_id] = {
                "s_min": s_min_ext.astype(np.float32),
                "s_max": s_max_ext.astype(np.float32),
                "start_idx": int(extended_start),
                "end_idx": int(extended_end),
                "collision_start_idx": int(start_idx),
                "collision_end_idx": int(end_idx),
                "collision_mask": collision_mask.astype(np.bool_),
                "time_indices": extended_indices.astype(np.int32),
                "time_distance_steps": time_distance_steps.astype(np.float32),
                "time_extension_steps_before": int(pad_before),
                "time_extension_steps_after": int(pad_after),
                "space_dilation": float(space_dilation),
                "action_type": action_type,
                "actor_type": actor_type,
                "importance": float(importance),
                "cost_scale": float(cost_scale),
            }

        return out

    def build_costmap_from_bands(
        self,
        t: np.ndarray,
        s: np.ndarray,
        bands_by_id: Dict[int, Dict[str, np.ndarray]],
        *,
        collision_cost: float = 1.0,
        yield_decay_exponent: float = 0.9,
        watch_out_decay_exponent: float = 1.2,
        default_decay_exponent: float = 1.1,
    ) -> np.ndarray:
        """
        Rasterize collision bands (per-actor) into a floating cost map cost[k,i].

        Inputs
        ------
        t : (K,) float32
            Global time grid for the planner. Indices 0..K-1.
        s : (S,) float32
            Global arclength grid for the planner. Indices 0..S-1.
        bands_by_id : dict
            For each actor id:
            {
                "s_min": (L,) float32,
                "s_max": (L,) float32,
                "start_idx": int,
                "end_idx": int,
                "collision_start_idx": int,
                "collision_end_idx": int,
                "collision_mask": (L,) bool,
                "time_indices": (L,) int,
                "time_distance_steps": (L,) float32,
                "time_extension_steps_before": int,
                "time_extension_steps_after": int,
                "space_dilation": float,
                "action_type": Optional[str],
                "actor_type": Optional[str],
                "importance": float,
                "cost_scale": float,
            }
            where L = end_idx - start_idx and arrays are aligned to that subrange.

        Output
        ------
        cost : (K, S) float32
            Non-negative costs for each (t_k, s_i) cell. Occupied cells receive the
            highest cost, while extension regions receive a decaying penalty.

        Notes
        -----
        - This function assumes the collision arrays are already trimmed to the interval
        [start_idx, end_idx] (as `compute_vehicle_s_bands` does).
        - If a band’s interval partially falls outside the planner’s [0, K-1] time window,
        it is clipped gracefully.
        - The main collision interval is treated as a hard cost while the
        action-driven time extension and spatial dilation smoothly decay towards
        zero cost at their limits.
        - Decay exponents are derived from the action type to shape how quickly the
        time-dependent decay falls off on either side of the collision window.
        """
        t = t.reshape(-1)
        s = s.reshape(-1)
        K, S = t.size, s.size

        cost = np.zeros((K, S), dtype=np.float32)
        if not bands_by_id:
            return cost

        decay_exponent_lookup = {
            "yield_for": float(yield_decay_exponent),
            "watch_out_for": float(watch_out_decay_exponent),
        }

        for actor_id, band in bands_by_id.items():
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
                    f"actor_id {actor_id}: expected len(s_min)==len(s_max)==end_idx-start_idx+1 "
                    f"({L}), got {smin_local.size} and {smax_local.size}"
                )

            # Clip the interval to the planner horizon [0, K-1]
            k0 = max(0, k_start)
            k1 = min(K - 1, k_end)
            if k1 < k0:
                continue  # completely outside

            span = (k1 - k0)             # how many rows to paint
            offset = k0 - k_start
            idx_hi = offset + span

            smin_slice = smin_local[offset:idx_hi]
            smax_slice = smax_local[offset:idx_hi]

            # Convert s-intervals to column index ranges on the global S grid
            j_lo = np.floor(smin_slice / self.st_grid_spec.ds)
            j_hi = np.floor(smax_slice / self.st_grid_spec.ds)

            j_lo = np.clip(j_lo, 0, S - 1).astype(np.int32)
            j_hi = np.clip(j_hi, 0, S - 1).astype(np.int32)

            action_type = band.get("action_type")
            collision_mask = band.get("collision_mask")
            time_indices = band.get("time_indices")
            time_distances = band.get("time_distance_steps")
            space_dilation = float(band.get("space_dilation", 0.0))
            collision_start_idx = int(band.get("collision_start_idx", k_start))
            collision_end_idx = int(band.get("collision_end_idx", k_end))
            ext_before = int(band.get("time_extension_steps_before", 0))
            ext_after = int(band.get("time_extension_steps_after", 0))
            cost_scale = float(band.get("cost_scale", 1.0))
            importance = float(band.get("importance", 0.0))
            actor_type = band.get("actor_type")
            decay_exponent = decay_exponent_lookup.get(action_type, float(default_decay_exponent))
            decay_exponent = float(max(decay_exponent - 0.2 * importance, 1e-6))
            actual_before = max(0, min(ext_before, collision_start_idx - k0))
            actual_after = max(0, min(ext_after, k1 - collision_end_idx))
            if actual_before > 0:
                before_norm_denom = float(max(actual_before - 1, 1))
            else:
                before_norm_denom = 1.0
            if actual_after > 0:
                after_norm_denom = float(max(actual_after - 1, 1))
            else:
                after_norm_denom = 1.0

            for r in range(span):
                lo = int(j_lo[r])
                hi = int(j_hi[r])
                if hi < lo:
                    continue

                s_values = s[lo:hi + 1]
                s_min_val = float(smin_slice[r])
                s_max_val = float(smax_slice[r])

                s_min_dil = s_min_val - space_dilation
                s_max_dil = s_max_val + space_dilation

                j_lo_dil = int(np.clip(np.floor(s_min_dil / self.st_grid_spec.ds), 0, S - 1))
                j_hi_dil = int(np.clip(np.floor(s_max_dil / self.st_grid_spec.ds), 0, S - 1))
                if j_hi_dil < j_lo_dil:
                    continue

                # Update with dilated indices if they extend beyond the initial
                # discretisation.
                if j_lo_dil < lo or j_hi_dil > hi:
                    lo = min(lo, j_lo_dil)
                    hi = max(hi, j_hi_dil)
                    s_values = s[lo:hi + 1]

                # Collision vs extension handling
                is_collision = True
                if collision_mask is not None and time_indices is not None:
                    idx = offset + r
                    if 0 <= idx < collision_mask.size:
                        is_collision = bool(collision_mask[idx])

                if is_collision:
                    base_cost = collision_cost * cost_scale
                else:
                    if action_type is None or time_distances is None or time_indices is None:
                        continue

                    idx = offset + r
                    if not (0 <= idx < time_distances.size and 0 <= idx < time_indices.size):
                        continue

                    time_distance = float(time_distances[idx])
                    if time_distance <= 0.0:
                        base_cost = collision_cost * cost_scale
                    else:
                        time_idx_val = int(time_indices[idx])
                        if time_idx_val < collision_start_idx:
                            denom = actual_before
                            norm_denom = before_norm_denom
                        else:
                            denom = actual_after
                            norm_denom = after_norm_denom

                        if denom <= 0:
                            continue

                        if denom == 1:
                            # Only a single extension step → treat as immediate drop to zero.
                            norm = float(time_distance >= 1.0)
                        else:
                            norm = (time_distance - 1.0) / norm_denom

                        norm = np.clip(norm, 0.0, 1.0)
                        time_decay = (1.0 - norm) ** decay_exponent
                        if time_decay <= 0.0:
                            continue
                        base_cost = collision_cost * cost_scale * time_decay

                # Apply spatial decay outside the true collision band.
                s_min_collision = s_min_val
                s_max_collision = s_max_val

                s_values = s[lo:hi + 1]
                inside_collision = (s_values >= s_min_collision) & (s_values <= s_max_collision)

                if space_dilation <= 1e-6:
                    # No dilation → all cells inside collision band take the base cost.
                    row_costs = np.where(inside_collision, base_cost, 0.0)
                else:
                    lower_dist = np.clip(s_min_collision - s_values, a_min=0.0, a_max=None)
                    upper_dist = np.clip(s_values - s_max_collision, a_min=0.0, a_max=None)
                    dist = lower_dist + upper_dist
                    space_decay = 1.0 - np.clip(dist / space_dilation, 0.0, 1.0)
                    row_costs = base_cost * space_decay
                    row_costs = np.where(inside_collision, base_cost, row_costs)

                if np.all(row_costs <= 0.0):
                    continue

                row_slice = slice(lo, hi + 1)
                cost[k0 + r, row_slice] = np.maximum(cost[k0 + r, row_slice], row_costs.astype(np.float32))

        return cost
