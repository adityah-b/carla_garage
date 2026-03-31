import carla
import numpy as np

from typing import Dict, Optional, Tuple, List

from config import GlobalConfig
from dataclasses import dataclass

from privileged_route_planner import PlannerState
from team_code.scene_descriptor.scene_descriptor import SceneData
from team_code.actor_prediction.motion_prediction import PredictionData

from actor_prediction.collision_checker import CollisionInterval, LaneOverlapInterval
from team_code.local_planner.directive_profiles import DilationProfile, resolve_dilation_profile

@dataclass
class STMaps:
    S_arr: np.ndarray
    T_arr: np.ndarray
    occupancy_map: np.ndarray
    cost_map: np.ndarray

class STOccupancyGrid:
    def __init__(self, config : GlobalConfig, st_grid_spec) -> None:
        self.config = config
        self.st_grid_spec = st_grid_spec

        # Store upstream prediction frequency
        self.prediction_frequency = self.config.prediction_frequency

        # Create s-T grid
        self.T_arr, self.S_arr = self.make_grids(st_grid_spec)

        # Store grid dimensions
        self.T_len = self.T_arr.size
        self.S_len = self.S_arr.size

        # Envelope cache (reused unless slope changes)
        self._envelope_slope: Optional[float] = None
        self._envelope_mask: Optional[np.ndarray] = None

        # Cost-shaping: resolved per-actor via directive_profiles module

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
        planner_state : PlannerState,
        scene_data : SceneData,
        prediction_data : PredictionData,
        all_conditions : Dict[int, Tuple] = {},
        s_bounds : Optional[Tuple[float, float]] = None,
    ) -> STMaps:
        max_speed = scene_data.traffic_data.speed_limit

        # ---------------- 1) compute per-actor s-bands ----------------
        tbb_bands, ebb_bands = self.compute_actor_s_bands(
            planner_state=planner_state,
            scene_data=scene_data,
            prediction_data=prediction_data,
            all_conditions=all_conditions,
            s_bounds=s_bounds,
        )

        # ---------------- 2) hard occupancy from TBBs (no dilation) ----------------
        occupancy_map = self.build_occupancy_from_bands(self.T_arr, self.S_arr, tbb_bands)

        # 3) Soft cost strictly from EBBs (now safely scaled down)
        # TODO: UPDATE THIS FIX THE COST MAP CONSTRUCTION
        cost_map = self.build_costmap_from_bands(self.T_arr, self.S_arr, ebb_bands, collision_cost=180.0)

        # --- ENFORCE HIERARCHY 1: True Collisions ---
        # Hard blocks (TBBs) must act as infinite cost walls to the Dijkstra search.
        # This overwrites any overlapping EBB soft costs with np.inf.
        cost_map[occupancy_map > 0] = self.st_grid_spec.collision_cost

        # 4) Penalise cells outside kinematic envelope
        valid = self.get_envelope_mask(max_speed)

        # --- ENFORCE HIERARCHY 2: Speed Limits ---
        # Do NOT hard-block (~valid) in the occupancy_map.
        # Apply the envelope cost to the cost_map, but use np.maximum so we
        # don't accidentally overwrite the np.inf collision walls we just built.
        env_cost = self.config.st_grid_spec.envelope_cost
        cost_map[~valid] = np.maximum(cost_map[~valid], env_cost)

        return STMaps(
            S_arr=self.S_arr,
            T_arr=self.T_arr,
            occupancy_map=occupancy_map,
            cost_map=cost_map,
        )

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

    def _build_s_bands_from_occupancies(
        self,
        frame_occupancies: Dict[int, Tuple[int, int]],
        t_start: int,
        t_end: int,
        route_bboxes: List[carla.BoundingBox],
        s_route: np.ndarray,
        ego_s_max: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project route-bbox occupancy indices to s-coordinates over [t_start, t_end]."""
        T = t_end - t_start + 1
        s_all = np.full((T, 2), np.inf, dtype=np.float32)
        for t_idx, (bb_start, bb_end) in frame_occupancies.items():
            t_rel = t_idx - t_start
            if not (0 <= t_rel < T):
                continue
            rs = self.config.bb_route_idx_to_dense_route_idx(bb_start)
            re = self.config.bb_route_idx_to_dense_route_idx(bb_end)

            rs = min(len(s_route) - 1, rs)
            re = min(len(s_route) - 1, re)

            s_min_val = s_route[rs] - route_bboxes[bb_start].extent.x
            s_max_val = s_route[re] + route_bboxes[bb_end].extent.x
            if s_min_val < ego_s_max < s_max_val and t_rel == 0:
                s_min_val = ego_s_max + 0.1
            s_all[t_rel, 0] = s_min_val
            s_all[t_rel, 1] = s_max_val
        return s_all[:, 0], s_all[:, 1]

    def _filter_bands_by_s_bounds(
        self,
        s_min: np.ndarray,
        s_max: np.ndarray,
        time_indices: np.ndarray,
        s_bounds: Optional[Tuple[float, float]],
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Trim to the contiguous valid window within s_bounds. Returns None if fully out of range."""
        valid_mask = np.ones(len(s_min), dtype=bool)
        if s_bounds is not None:
            s_bound_min, s_bound_max = s_bounds
            valid_mask = ~((s_max < s_bound_min) | (s_min > s_bound_max))
        if not np.any(valid_mask):
            return None
        locs = np.where(valid_mask)[0]
        first, last = locs[0], locs[-1]
        return s_min[first:last + 1], s_max[first:last + 1], time_indices[first:last + 1]

    def _make_s_band_entry(
        self,
        s_min: np.ndarray,
        s_max: np.ndarray,
        time_indices: np.ndarray,
        profile: DilationProfile,
        directive: str = "default",
    ) -> Dict:
        return {
            "prediction_frequency": self.prediction_frequency,
            "s_min": s_min,
            "s_max": s_max,
            "collision_start_idx": int(time_indices[0]),
            "collision_end_idx": int(time_indices[-1]),
            "action_type": directive,
            "time_pad_before": profile.dt_before,
            "time_pad_after": profile.dt_after,
            "space_dilation_before": profile.ds_st_before,
            "space_dilation_after": profile.ds_st_after,
            "cost_scale": profile.cost_scale,
            "decay_exponent": profile.decay_st,
        }

    def compute_actor_s_bands(
        self,
        planner_state : PlannerState,
        scene_data : SceneData,
        prediction_data : PredictionData,
        *,
        all_conditions: Dict[int, Tuple] = {},
        s_bounds: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[int, Dict[str, np.ndarray]]]:
        """
        For each actor compute s-bands from TBB and EBB route overlap data.

        Returns
        -------
        tbb_bands : Dict[actor_id → band]
            Bands strictly from physical bounding boxes (TBB).
            Dilation is 0. Used for hard occupancy.
        cost_bands : Dict[actor_id → band]
            Bands from extended bounding boxes (EBB).
            Used purely for soft cost (decay padding around TBB).
        """
        tbb_bands: Dict[int, Dict[str, np.ndarray]] = {}
        cost_bands: Dict[int, Dict[str, np.ndarray]] = {}

        route_index = planner_state.route_index
        s_route = planner_state.s_route[route_index:] - planner_state.s_route[route_index]
        ego_s_max = self.config.ego_extent_x

        all_actor_overlaps = prediction_data.all_actor_overlaps

        _ZERO_PROFILE = DilationProfile()

        for actor_id, overlap_interval in all_actor_overlaps.items():
            if actor_id in all_conditions:
                directive, actor_type, traffic_type, priority = all_conditions[actor_id]
            else:
                directive, actor_type, traffic_type, priority = "default", "default", "default", "medium"

            profile = resolve_dilation_profile(directive, actor_type, traffic_type, priority)

            route_bboxes: List[carla.BoundingBox] = overlap_interval.route_subset_bboxes
            tbb_occ = overlap_interval.tbb_frame_occupancies
            ebb_occ = overlap_interval.ebb_frame_occupancies

            # ---------------------------------------------------------
            # 1. TBB: Hard Occupancy (STRICTLY NO DILATION)
            # ---------------------------------------------------------
            result = None
            if len(tbb_occ) > 0:
                tbb_times = sorted(tbb_occ.keys())
                tbb_t_start, tbb_t_end = tbb_times[0], tbb_times[-1]
                tbb_time_indices = np.arange(tbb_t_start, tbb_t_end + 1, dtype=np.int32)

                tbb_s_min, tbb_s_max = self._build_s_bands_from_occupancies(
                    tbb_occ, tbb_t_start, tbb_t_end, route_bboxes, s_route, ego_s_max
                )
                result = self._filter_bands_by_s_bounds(tbb_s_min, tbb_s_max, tbb_time_indices, s_bounds)
                if result is not None:
                    tbb_s_min, tbb_s_max, tbb_time_indices = result
                    tbb_bands[actor_id] = self._make_s_band_entry(
                        tbb_s_min, tbb_s_max, tbb_time_indices,
                        profile=_ZERO_PROFILE, directive=directive,
                    )

            # ---------------------------------------------------------
            # 2. EBB: Soft Cost (EBB baseline + directive ST deltas)
            # ---------------------------------------------------------
            if len(ebb_occ) > 0:
                ebb_t_start = overlap_interval.time_start_idx
                ebb_t_end = overlap_interval.time_end_idx
                ebb_time_indices = np.arange(ebb_t_start, ebb_t_end + 1, dtype=np.int32)

                ebb_s_min, ebb_s_max = self._build_s_bands_from_occupancies(
                    ebb_occ, ebb_t_start, ebb_t_end, route_bboxes, s_route, ego_s_max
                )
                ebb_result = self._filter_bands_by_s_bounds(ebb_s_min, ebb_s_max, ebb_time_indices, s_bounds)
                if ebb_result is not None:
                    ebb_s_min, ebb_s_max, ebb_time_indices = ebb_result
                    cost_bands[actor_id] = self._make_s_band_entry(
                        ebb_s_min, ebb_s_max, ebb_time_indices,
                        profile=profile, directive=directive,
                    )
            elif len(tbb_occ) > 0 and result is not None:
                # Fallback: no EBB exists — use TBB + directive profile dilation
                cost_bands[actor_id] = self._make_s_band_entry(
                    tbb_s_min, tbb_s_max, tbb_time_indices,
                    profile=profile, directive=directive,
                )

        return tbb_bands, cost_bands

    def get_frenet_s_bounds(
        self,
        bbox_corners_xy_N: np.ndarray,
        route_xy: np.ndarray,
        route_s: np.ndarray,
        route_yaws: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized projection of N bounding boxes (each with 4 corners) onto a reference path.

        Args:
            corners_xy_N: (N, 4, 2) array of [x, y] coordinates for bounding box corners over time.
            route_x, route_y: (M,) arrays of the reference path coordinates (sliced for efficiency).
            route_s: (M,) array of accumulated s-values along the path.
            route_theta: (M,) array of heading angles (in radians) along the path.

        Returns:
            s_min, s_max: Two (N,) arrays containing the min and max s-values per timestep.
        """
        # 1. Expand dimensions for broadcasting
        # corners: (N, 4, 1, 2) | route: (1, 1, M, 2)
        corners_expanded = bbox_corners_xy_N[:, :, np.newaxis, :]
        route_expanded = route_xy[np.newaxis, np.newaxis, :, :]

        # 2. Compute distances from all corners at all times to all route points
        # Resulting shape: (N, 4, M)
        diff = corners_expanded - route_expanded
        dists_sq = np.sum(diff * diff, axis=3)

        # 3. Find the index of the nearest route point for each corner at each time
        # Resulting shape: (N, 4)
        nearest_idx = np.argmin(dists_sq, axis=2)

        # 4. Extract reference data using advanced indexing
        ref_x = route_xy[nearest_idx, 0]         # (N, 4)
        ref_y = route_xy[nearest_idx, 1]         # (N, 4)
        ref_s = route_s[nearest_idx]         # (N, 4)
        ref_theta = route_yaws[nearest_idx] # (N, 4)

        # 5. Compute displacement vectors
        dx = bbox_corners_xy_N[:, :, 0] - ref_x   # (N, 4)
        dy = bbox_corners_xy_N[:, :, 1] - ref_y   # (N, 4)

        # 6. normal vectors at the reference points
        t_x, t_y = np.cos(ref_theta), np.sin(ref_theta)
        n_x, n_y = -np.sin(ref_theta), np.cos(ref_theta)

        # 7. Dot products to get Delta S and exact D
        delta_s = dx * t_x + dy * t_y        # (N, 4)
        d_vals  = dx * n_x + dy * n_y        # (N, 4)

        # 8. Calculate absolute S for all corners
        s_corners = ref_s + delta_s          # (N, 4)

        s_corners = np.clip(s_corners, route_s[0], route_s[-1])

        # 9. Reduce to min and max along the corner axis (axis=1)
        s_min = np.min(s_corners, axis=1)    # (N,)
        s_max = np.max(s_corners, axis=1)    # (N,)

        d_min = np.min(d_vals, axis=1)
        d_max = np.max(d_vals, axis=1)

        return s_min, s_max, d_min, d_max

    # def compute_actor_s_bands(
    #     self,
    #     planner_state : PlannerState,
    #     scene_data : SceneData,
    #     prediction_data : PredictionData,
    #     *,
    #     all_conditions: Dict[int, Tuple] = {},
    #     s_bounds: Optional[Tuple[float, float]] = None,
    # ) -> Dict[int, Dict[str, np.ndarray]]:
    #     """
    #     For each actor, project its four OBB corners per time to s, then
    #     take min/max across corners → s_min(t), s_max(t).

    #     `all_conditions` lets callers describe how to extend and weight the costs
    #     for specific actors using (action_type, actor_type, importance):
    #         - action_type ∈ {"yield_for", "watch_out_for"}
    #         - actor_type  ∈ {"vehicle", "cyclist", "ped"}
    #         - importance  ∈ [0.0, 1.0]

    #     "yield_for" emphasises waiting until the actor passes the conflict region
    #     by padding the time dimension primarily before the collision interval.
    #     "watch_out_for" builds symmetric time padding and a larger spatial buffer
    #     around the collision band. Actor type and importance scale the collision
    #     cost and dilation to reflect perceived risk.
    #     """
    #     out: Dict[int, Dict[str, np.ndarray]] = {}

    #     route_index = planner_state.route_index
    #     route_pts = planner_state.route_points[route_index:]
    #     route_yaws = planner_state.rotation_angles[route_index:]
    #     s_route = planner_state.s_route[route_index:] - planner_state.s_route[route_index]

    #     # Get overlap intervals
    #     all_actor_overlaps = prediction_data.all_actor_overlaps

    #     # Project overlap intervals
    #     for actor_id, overlap_interval in all_actor_overlaps.items():
    #         if not overlap_interval.is_valid:
    #             continue

    #         # Get conditions for each actor
    #         if actor_id in all_conditions:
    #             action_type, actor_type, importance = all_conditions[actor_id]
    #         else:
    #             # action_type, actor_type, importance = "yield_for", "default", 1.0
    #             action_type, actor_type, importance = "default", "default", 1.0

    #         # Get space and time padding based on command and object type
    #         time_pad_before, time_pad_after = self.temporal_dilation_by_cmd[action_type]

    #         cost_scale = 1.0 + 0.75 * importance

    #         # NOTE: Only using single overlap interval
    #         # Get overlap interval time indices
    #         t_start_idx = overlap_interval.time_start_idx
    #         t_end_idx = overlap_interval.time_end_idx

    #         # Create time index array
    #         time_indices = np.arange(t_start_idx, t_end_idx + 1, dtype=np.int32)

    #         # Get route and overlapping bounding box data
    #         route_start_idx = overlap_interval.space_start_idx * 20
    #         route_end_idx = overlap_interval.space_end_idx * 20
    #         route_slice = slice(route_start_idx, route_end_idx + 1)

    #         actor_ebb_corners_N = overlap_interval.actor_overlapping_ebb_corners
    #         print(f'\n\nFRENET TRANSFORM')
    #         print(f'\tACTOR ID: {actor_id}, BB_IDX: {overlap_interval.space_start_idx}, ROUTE_START_IDX: {route_start_idx}, S_ROUTE[0]: {s_route[route_start_idx]}')
    #         s_min, s_max, d_min, d_max = self.get_frenet_s_bounds(
    #             bbox_corners_xy_N=actor_ebb_corners_N,
    #             route_xy=route_pts[route_slice, :2],
    #             route_s=s_route[route_slice],
    #             route_yaws=route_yaws[route_slice],
    #         )

    #         # print(f'\n\nS_MIN: {s_min}, \nS_MAX: {s_max}, \nD_MIN: {d_min}, \nD_MAX: {d_max}')

    #         D_THRESHOLD = self.config.ego_extent_y * 1.1
    #         lateral_threat_mask = (d_min <= D_THRESHOLD) & (d_max >= -D_THRESHOLD)

    #         # 2. Define S-Bounds Mask (with corrected parenthesis and OR logic)
    #         if s_bounds is not None:
    #             s_bound_min, s_bound_max = s_bounds
    #             # Mask is TRUE if the vehicle is strictly outside the bounds
    #             s_out_of_bounds = (s_max < s_bound_min) | (s_min > s_bound_max)
    #             s_threat_mask = ~s_out_of_bounds
    #             print(f'\n\nACTOR ID: {actor_id}, S_MIN: {s_min}, S_BOUND_MIN: {s_bound_min}')
    #         else:
    #             s_threat_mask = np.ones_like(lateral_threat_mask, dtype=bool)

    #         # 3. Combine Masks: A threat only exists if BOTH S and D overlap
    #         valid_mask = lateral_threat_mask & s_threat_mask

    #         # 4. Handle vehicles that are completely harmless
    #         if not np.any(valid_mask):
    #             continue # Skip adding this actor to the output dictionary

    #         # 5. Crop arrays to the continuous valid time window
    #         # This prevents creating gaps in the middle of your time arrays,
    #         # which would break your downstream ST grid builder.
    #         valid_indices = np.where(valid_mask)[0]
    #         first_idx = valid_indices[0]
    #         last_idx = valid_indices[-1]

    #         s_min = s_min[first_idx:last_idx + 1]
    #         s_max = s_max[first_idx:last_idx + 1]
    #         time_indices = time_indices[first_idx:last_idx + 1]

    #         # Recompute the start and end collision indices for the valid window
    #         t_start_idx = time_indices[0]
    #         t_end_idx = time_indices[-1]

    #         out[actor_id] = {
    #             # Prediction frequency
    #             "prediction_frequency" : self.prediction_frequency,

    #             # Collision band extents over time
    #             "s_min": s_min,
    #             "s_max": s_max,

    #             # Use the new filtered time indices
    #             "collision_start_idx": t_start_idx,
    #             "collision_end_idx": t_end_idx,

    #             # Command directive parameters
    #             "action_type": action_type,
    #             "actor_type": actor_type,
    #             "importance": importance,
    #             "time_pad_before" : time_pad_before,
    #             "time_pad_after" : time_pad_after,
    #             "space_dilation" : 0.0,
    #             "cost_scale": cost_scale,
    #         }

    #     return out

    # def build_costmap_from_bands(
    #     self,
    #     t: np.ndarray,
    #     s: np.ndarray,
    #     bands_by_id: Dict[int, Dict[str, np.ndarray]],
    #     *,
    #     collision_cost: float = 255.0,
    #     yield_decay_exponent: float = 0.5,
    #     watch_out_decay_exponent: float = 1.2,
    #     default_decay_exponent: float = 1.1,
    # ) -> np.ndarray:
    #     """
    #     Rasterize collision bands (per-actor) into a floating cost map cost[k,i].

    #     Inputs
    #     ------
    #     t : (K,) float32
    #         Global time grid for the planner. Indices 0..K-1.
    #     s : (S,) float32
    #         Global arclength grid for the planner. Indices 0..S-1.
    #     bands_by_id : dict
    #         For each actor id:
    #         {
    #             "s_min": (L,) float32,
    #             "s_max": (L,) float32,
    #             "start_idx": int,
    #             "end_idx": int,
    #             "collision_start_idx": int,
    #             "collision_end_idx": int,
    #             "collision_mask": (L,) bool,
    #             "time_indices": (L,) int,
    #             "time_distance_steps": (L,) float32,
    #             "time_extension_steps_before": int,
    #             "time_extension_steps_after": int,
    #             "space_dilation": float,
    #             "action_type": Optional[str],
    #             "actor_type": Optional[str],
    #             "importance": float,
    #             "cost_scale": float,
    #         }
    #         where L = end_idx - start_idx and arrays are aligned to that subrange.

    #     Output
    #     ------
    #     cost : (K, S) float32
    #         Non-negative costs for each (t_k, s_i) cell. Occupied cells receive the
    #         highest cost, while extension regions receive a decaying penalty.

    #     Notes
    #     -----
    #     - This function assumes the collision arrays are already trimmed to the interval
    #     [start_idx, end_idx] (as `compute_vehicle_s_bands` does).
    #     - If a band’s interval partially falls outside the planner’s [0, K-1] time window,
    #     it is clipped gracefully.
    #     - The main collision interval is treated as a hard cost while the
    #     action-driven time extension and spatial dilation smoothly decay towards
    #     zero cost at their limits.
    #     - Decay exponents are derived from the action type to shape how quickly the
    #     time-dependent decay falls off on either side of the collision window.
    #     """
    #     t = t.reshape(-1)
    #     s = s.reshape(-1)
    #     K, S = t.size, s.size

    #     cost = np.zeros((K, S), dtype=np.float32)
    #     if not bands_by_id:
    #         return cost

    #     decay_exponent_lookup = {
    #         "yield_for": float(yield_decay_exponent),
    #         "watch_out_for": float(watch_out_decay_exponent),
    #     }

    #     for actor_id, band in bands_by_id.items():
    #         smin_local = band["s_min"]
    #         smax_local = band["s_max"]

    #         k_start = int(band["start_idx"])  # inclusive
    #         k_end   = int(band["end_idx"])    # exclusive

    #         # Length checks and early outs
    #         L = k_end - k_start
    #         if L <= 0 or smin_local.size == 0 or smax_local.size == 0:
    #             continue
    #         if smin_local.size != L or smax_local.size != L:
    #             raise ValueError(
    #                 f"actor_id {actor_id}: expected len(s_min)==len(s_max)==end_idx-start_idx+1 "
    #                 f"({L}), got {smin_local.size} and {smax_local.size}"
    #             )

    #         # Clip the interval to the planner horizon [0, K-1]
    #         k0 = max(0, k_start)
    #         k1 = min(K - 1, k_end)
    #         if k1 < k0:
    #             continue  # completely outside

    #         span = (k1 - k0)             # how many rows to paint
    #         offset = k0 - k_start
    #         idx_hi = offset + span

    #         smin_slice = smin_local[offset:idx_hi]
    #         smax_slice = smax_local[offset:idx_hi]

    #         # Convert s-intervals to column index ranges on the global S grid
    #         j_lo = np.floor(smin_slice / self.st_grid_spec.ds)
    #         j_hi = np.floor(smax_slice / self.st_grid_spec.ds)

    #         j_lo = np.clip(j_lo, 0, S - 1).astype(np.int32)
    #         j_hi = np.clip(j_hi, 0, S - 1).astype(np.int32)

    #         action_type = band.get("action_type")
    #         collision_mask = band.get("collision_mask")
    #         time_indices = band.get("time_indices")
    #         time_distances = band.get("time_distance_steps")
    #         space_dilation = float(band.get("space_dilation", 0.0))
    #         collision_start_idx = int(band.get("collision_start_idx", k_start))
    #         collision_end_idx = int(band.get("collision_end_idx", k_end))
    #         ext_before = int(band.get("time_extension_steps_before", 0))
    #         ext_after = int(band.get("time_extension_steps_after", 0))
    #         cost_scale = float(band.get("cost_scale", 1.0))
    #         importance = float(band.get("importance", 0.0))
    #         actor_type = band.get("actor_type")
    #         decay_exponent = decay_exponent_lookup.get(action_type, float(default_decay_exponent))
    #         decay_exponent = float(max(decay_exponent - 0.2 * importance, 1e-6))
    #         actual_before = max(0, min(ext_before, collision_start_idx - k0))
    #         actual_after = max(0, min(ext_after, k1 - collision_end_idx))
    #         if actual_before > 0:
    #             before_norm_denom = float(max(actual_before - 1, 1))
    #         else:
    #             before_norm_denom = 1.0
    #         if actual_after > 0:
    #             after_norm_denom = float(max(actual_after - 1, 1))
    #         else:
    #             after_norm_denom = 1.0

    #         for r in range(span):
    #             lo = int(j_lo[r])
    #             hi = int(j_hi[r])
    #             if hi < lo:
    #                 continue

    #             s_values = s[lo:hi + 1]
    #             s_min_val = float(smin_slice[r])
    #             s_max_val = float(smax_slice[r])

    #             s_min_dil = s_min_val - space_dilation
    #             s_max_dil = s_max_val + space_dilation

    #             j_lo_dil = int(np.clip(np.floor(s_min_dil / self.st_grid_spec.ds), 0, S - 1))
    #             j_hi_dil = int(np.clip(np.floor(s_max_dil / self.st_grid_spec.ds), 0, S - 1))
    #             if j_hi_dil < j_lo_dil:
    #                 continue

    #             # Update with dilated indices if they extend beyond the initial
    #             # discretisation.
    #             if j_lo_dil < lo or j_hi_dil > hi:
    #                 lo = min(lo, j_lo_dil)
    #                 hi = max(hi, j_hi_dil)
    #                 s_values = s[lo:hi + 1]

    #             # Collision vs extension handling
    #             is_collision = True
    #             if collision_mask is not None and time_indices is not None:
    #                 idx = offset + r
    #                 if 0 <= idx < collision_mask.size:
    #                     is_collision = bool(collision_mask[idx])

    #             if is_collision:
    #                 base_cost = collision_cost * cost_scale
    #             else:
    #                 if action_type is None or time_distances is None or time_indices is None:
    #                     continue

    #                 idx = offset + r
    #                 if not (0 <= idx < time_distances.size and 0 <= idx < time_indices.size):
    #                     continue

    #                 time_distance = float(time_distances[idx])
    #                 if time_distance <= 0.0:
    #                     base_cost = collision_cost * cost_scale
    #                 else:
    #                     time_idx_val = int(time_indices[idx])
    #                     if time_idx_val < collision_start_idx:
    #                         denom = actual_before
    #                         norm_denom = before_norm_denom
    #                     else:
    #                         denom = actual_after
    #                         norm_denom = after_norm_denom

    #                     if denom <= 0:
    #                         continue

    #                     if denom == 1:
    #                         # Only a single extension step → treat as immediate drop to zero.
    #                         norm = float(time_distance >= 1.0)
    #                     else:
    #                         norm = (time_distance - 1.0) / norm_denom

    #                     norm = np.clip(norm, 0.0, 1.0)
    #                     time_decay = (1.0 - norm) ** decay_exponent
    #                     if time_decay <= 0.0:
    #                         continue
    #                     base_cost = collision_cost * cost_scale * time_decay

    #             # Apply spatial decay outside the true collision band.
    #             s_min_collision = s_min_val
    #             s_max_collision = s_max_val

    #             s_values = s[lo:hi + 1]
    #             inside_collision = (s_values >= s_min_collision) & (s_values <= s_max_collision)

    #             if space_dilation <= 1e-6:
    #                 # No dilation → all cells inside collision band take the base cost.
    #                 row_costs = np.where(inside_collision, base_cost, 0.0)
    #             else:
    #                 lower_dist = np.clip(s_min_collision - s_values, a_min=0.0, a_max=None)
    #                 upper_dist = np.clip(s_values - s_max_collision, a_min=0.0, a_max=None)
    #                 dist = lower_dist + upper_dist
    #                 space_decay = 1.0 - np.clip(dist / space_dilation, 0.0, 1.0)
    #                 row_costs = base_cost * space_decay
    #                 row_costs = np.where(inside_collision, base_cost, row_costs)

    #             if np.all(row_costs <= 0.0):
    #                 continue

    #             row_slice = slice(lo, hi + 1)
    #             cost[k0 + r, row_slice] = np.maximum(cost[k0 + r, row_slice], row_costs.astype(np.float32))

    #     return cost

    def build_occupancy_from_bands(
        self,
        t: np.ndarray,
        s: np.ndarray,
        tbb_bands_by_id: Dict[int, Dict[str, np.ndarray]],
    ) -> np.ndarray:
        """
        Rasterize TBB bands into a binary hard-occupancy grid.

        Cells covered by the true bounding-box s-interval at each timestep are
        set to ``collision_cost``; all other cells remain 0.  No spatial dilation
        or temporal decay is applied — the core footprint is strictly blocked.
        """
        t = t.reshape(-1)
        s = s.reshape(-1)
        K, S = t.size, s.size

        collision_cost = self.config.st_grid_spec.collision_cost
        occ = np.zeros((K, S), dtype=np.float32)

        if not tbb_bands_by_id:
            return occ

        ds = float(self.st_grid_spec.ds)

        for actor_id, band in tbb_bands_by_id.items():
            s_min_local = band["s_min"]
            s_max_local = band["s_max"]
            if s_min_local.size == 0:
                continue

            k_start = band["collision_start_idx"]
            k_end   = band["collision_end_idx"]
            k0 = max(0, k_start)
            k1 = min(K - 1, k_end)
            if k1 < k0:
                continue

            offset = k0 - k_start
            span   = k1 - k0 + 1
            smin_slice = s_min_local[offset : offset + span]
            smax_slice = s_max_local[offset : offset + span]

            j_lo = np.clip(np.floor(smin_slice / ds), 0, S - 1).astype(np.int32)
            j_hi = np.clip(np.floor(smax_slice / ds), 0, S - 1).astype(np.int32)

            for r in range(span):
                if not (np.isfinite(smin_slice[r]) and np.isfinite(smax_slice[r])):
                    continue
                lo, hi = int(j_lo[r]), int(j_hi[r])
                if hi >= lo:
                    occ[k0 + r, lo : hi + 1] = collision_cost

        return occ

    def build_costmap_from_bands(
        self,
        t: np.ndarray,
        s: np.ndarray,
        bands_by_id: Dict[int, Dict[str, np.ndarray]],
        collision_cost : float,
    ) -> np.ndarray:
        """
        Global envelope + 1D (time-axis) distance-to-core mask.

        - Build a core mask for the exact (dilated) collision band over the true collision
        time interval only.
        - Build an envelope rectangle over an extended time window and global s-range.
        - Inside the envelope, compute per-column 1D distance in time to the nearest
        core-mask True cell, and map distance -> cost via an action-dependent decay exponent.
        - Core cells receive max cost.

        Assumption: planning and prediction time indices are aligned (row k corresponds to prediction index k).
        """
        t = t.reshape(-1)
        s = s.reshape(-1)
        K, S = t.size, s.size

        cost = np.zeros((K, S), dtype=np.float32)
        if not bands_by_id:
            return cost

        ds = float(self.st_grid_spec.ds)

        for actor_id, band in bands_by_id.items():
            s_min_local = band["s_min"]
            s_max_local = band["s_max"]

            # Core collision indices
            k_col_start = band["collision_start_idx"]
            k_col_end = band["collision_end_idx"]

            # --- NEW: Clamp temporal indices and crop spatial arrays to ST grid bounds ---
            if k_col_start >= K or k_col_end < 0:
                continue  # Prediction is entirely outside the ST planning horizon

            valid_start = max(0, -k_col_start)
            valid_end = min(len(s_min_local), K - k_col_start)

            # Crop the arrays so they match the clamped time interval
            s_min_local = s_min_local[valid_start:valid_end]
            s_max_local = s_max_local[valid_start:valid_end]

            # Clamp the time indices
            k_col_start = max(0, k_col_start)
            k_col_end = min(K - 1, k_col_end)
            # -----------------------------------------------------------------------------

            if s_min_local.size == 0 or s_max_local.size == 0:
                continue
            if not (s_min_local.size == s_max_local.size):
                raise ValueError(
                    f"actor_id {actor_id}: s_min/s_max must have same length; "
                    f"got {s_min_local.size}, {s_max_local.size}"
                )

            # Command directive parameters
            prediction_frequency = band["prediction_frequency"]

            cost_scale = band["cost_scale"]

            time_pad_before_s = band["time_pad_before"]
            time_pad_after_s = band["time_pad_after"]

            space_dilation_before = band["space_dilation_before"]
            space_dilation_after = band["space_dilation_after"]

            decay_exponent = band["decay_exponent"]

            # Cost temporal window in steps
            time_pad_before_steps = int(np.round(time_pad_before_s * prediction_frequency))
            time_pad_after_steps = int(np.round(time_pad_after_s * prediction_frequency))
            k_ext_start = max(0, k_col_start - time_pad_before_steps)
            k_ext_end = min(K - 1, k_col_end + time_pad_after_steps)

            # Get global envelope over collision region with asymmetric spatial dilation
            s_min_global = np.min(s_min_local) - space_dilation_before
            s_max_global = np.max(s_max_local) + space_dilation_after

            j_lo_global = np.clip(np.floor(s_min_global / ds), 0, S - 1).astype(np.int32)
            j_hi_global = np.clip(np.floor(s_max_global / ds), 0, S - 1).astype(np.int32)

            # Build envelope region
            ENV_ROWS = k_ext_end - k_ext_start + 1
            ENV_COLS = j_hi_global - j_lo_global + 1
            envelope_mask = np.zeros((ENV_ROWS, ENV_COLS), dtype=np.bool_)

            # Populate envelope mask with true collision region (asymmetric spatial dilation)
            s_min_dil = s_min_local - space_dilation_before
            s_max_dil = s_max_local + space_dilation_after

            j_lo_dil = np.clip(np.floor(s_min_dil / ds), j_lo_global, j_hi_global).astype(np.int32)
            j_hi_dil = np.clip(np.floor(s_max_dil / ds), j_lo_global, j_hi_global).astype(np.int32)

            col_idx_env = np.arange(ENV_COLS)[None, :]  # Shape (1, ENV_COLS)

            # 2. Map global dilation boundaries to local envelope coordinates
            lo_local = (j_lo_dil - j_lo_global)[:, None]  # Shape (COL_SPAN, 1)
            hi_local = (j_hi_dil - j_lo_global)[:, None]  # Shape (COL_SPAN, 1)

            # 3. Create the mask via broadcasting (No Python Loop)
            # A cell is True if its column index is between the local lo and hi boundaries
            core_mask_in_envelope = (col_idx_env >= lo_local) & (col_idx_env <= hi_local)

            # 4. Place the core mask into the larger envelope_mask
            # This handles the case where k_col_start/end might differ from k_ext_start/end
            COL_SPAN = k_col_end - k_col_start + 1
            rr_start = k_col_start - k_ext_start
            envelope_mask[rr_start : rr_start + COL_SPAN, :] = core_mask_in_envelope

            max_cost = float(collision_cost * cost_scale)

            mask = envelope_mask  # (ENV_ROWS, ENV_COLS) bool
            N, W = mask.shape
            idx = np.arange(N, dtype=np.int32)[:, None]  # (N,1)

            # Dist to nearest True in the past (left / up): dist_left
            left_idx = np.where(mask, idx, -N)                       # (N,W)
            last_true_left = np.maximum.accumulate(left_idx, axis=0) # (N,W)
            dist_left = (idx - last_true_left).astype(np.float32)    # (N,W)

            # Dist to nearest True in the future (right / down): dist_right
            right_idx = np.where(mask, idx, 2 * N)                                  # (N,W)
            last_true_right = np.minimum.accumulate(right_idx[::-1], axis=0)[::-1]  # (N,W)
            dist_right = (last_true_right - idx).astype(np.float32)                 # (N,W)
            before_denom = float(max(time_pad_before_steps, 1))
            after_denom  = float(max(time_pad_after_steps, 1))

            # Asymmetric: BEFORE uses dist_right; AFTER uses dist_left
            before_norm  = np.clip(dist_right / before_denom, 0.0, 1.0)
            after_norm   = np.clip(dist_left  / after_denom,  0.0, 1.0)

            before_cost = max_cost * (1.0 - before_norm) ** decay_exponent
            after_cost  = max_cost * (1.0 - after_norm)  ** decay_exponent

            env_cost = np.maximum(before_cost, after_cost).astype(np.float32)  # (ENV_ROWS, ENV_COLS)

            # Write the whole envelope block in one shot (max-composition across actors)
            row_slice = slice(k_ext_start, k_ext_end + 1)
            col_slice = slice(int(j_lo_global), int(j_hi_global) + 1)
            cost[row_slice, col_slice] = np.maximum(cost[row_slice, col_slice], env_cost)

            # # Ensure core is exactly max cost
            # cost[row_slice, col_slice][mask] = np.maximum(cost[row_slice, col_slice][mask], np.float32(max_cost))

        return cost