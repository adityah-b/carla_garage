import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from config import GlobalConfig
from privileged_route_planner import PlannerState
from team_code.scene_descriptor.scene_descriptor import SceneData
from team_code.actor_prediction.motion_prediction import PredictionData
from team_code.scene_analyzer.parsers.ego_plan_pydantic_models import *
from team_code.local_planner.directive_profiles import DilationProfile, resolve_dilation_profile

@dataclass
class SLMaps:
    S_arr : np.ndarray
    L_arr : np.ndarray

    L_ref : float
    target_lane_width : float

    occupancy_map: np.ndarray
    cost_map: np.ndarray

class SLOccupancyGrid:
    def __init__(self, config : GlobalConfig) -> None:
        self.config = config
        self.sl_grid_spec = config.sl_grid_spec

        # Create S-L grid
        self.S_arr, self.L_arr = self.make_grids(self.sl_grid_spec)

        # Store grid dimensions
        self.S_len = self.S_arr.size
        self.L_len = self.L_arr.size

        # Cost-shaping: resolved per-actor via directive_profiles module

    def make_grids(self, sl_grid_spec) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform S-L grids."""
        S_arr = np.arange(0.0, sl_grid_spec.S_max + 1e-6, sl_grid_spec.ds, dtype=np.float32)
        L_arr = np.arange(sl_grid_spec.L_min, sl_grid_spec.L_max + 1e-6, sl_grid_spec.dl, dtype=np.float32)
        return S_arr, L_arr

    def build_maps(
        self,
        planner_state : PlannerState,
        scene_data : SceneData,
        prediction_data : PredictionData,
        ego_plan : EgoPlan,
        all_conditions: Dict[int, Tuple] = {},
        s_bounds: Optional[Tuple[float, float]] = None,
    ) -> SLMaps:

        # Determine dynamic grid bounds from the current command + scene context.
        L_min, L_max, S_max = self.find_grid_boundaries(planner_state, scene_data, ego_plan)

        # Slice the global fixed-resolution arrays to the dynamic window.
        S_dyn = self.S_arr[self.S_arr <= S_max + 1e-6]
        L_dyn = self.L_arr[(self.L_arr >= L_min - 1e-6) & (self.L_arr <= L_max + 1e-6)]

        # Project all scene actors onto metric S-L bands along the planned route.
        bands = self.compute_actor_sl_bands(
            planner_state=planner_state,
            scene_data=scene_data,
            prediction_data=prediction_data,
            all_conditions=all_conditions,
            s_bounds=s_bounds,
        )

        # Rasterise bands onto the dynamic grid.
        occ_map, cost_obs = self.build_costmap_from_bands(S_dyn, L_dyn, bands_by_id=bands)

        # Build road-alignment cost: quadratic attraction toward the reference lane
        #    centre implied by the current action command.
        l_ref, target_lane_width, cost_road = self.apply_road_cost(planner_state, scene_data, ego_plan, L_dyn, S_dyn)

        # 6) Compose and return
        # cost_total = np.maximum(cost_obs, cost_road)
        cost_total = cost_obs + cost_road

        return SLMaps(
            S_arr=S_dyn,
            L_arr=L_dyn,
            L_ref=l_ref,
            target_lane_width=target_lane_width,
            occupancy_map=occ_map,
            cost_map=cost_total,
        )

    def get_frenet_bounds(
        self,
        bbox_corners_xy_N: np.ndarray,
        route_xy: np.ndarray,
        route_s: np.ndarray,
        route_yaws: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized projection of N bounding boxes (each with 4 corners) onto a reference path.

        Args:
            corners_xy_N: (N, 4, 2) array of [x, y] coordinates for bounding box corners over time.
            route_xy: (M, 2) array of the reference path XY coordinates (sliced for efficiency).
            route_s: (M,) array of accumulated s-values along the path.
            route_yaws: (M,) array of heading angles in degrees along the path.
                        Converted to radians internally before trig operations.

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
        ref_x = route_xy[nearest_idx, 0]                    # (N, 4)
        ref_y = route_xy[nearest_idx, 1]                    # (N, 4)
        ref_s = route_s[nearest_idx]                        # (N, 4)
        ref_theta = np.deg2rad(route_yaws[nearest_idx])     # (N, 4) degrees → radians

        # 5. Compute displacement vectors
        dx = bbox_corners_xy_N[:, :, 0] - ref_x   # (N, 4)
        dy = bbox_corners_xy_N[:, :, 1] - ref_y   # (N, 4)

        # 6. normal vectors at the reference points
        t_x, t_y = np.cos(ref_theta), np.sin(ref_theta)
        n_x, n_y = np.sin(ref_theta), -np.cos(ref_theta)

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

    def compute_actor_sl_bands(
        self,
        planner_state : PlannerState,
        scene_data : SceneData,
        prediction_data : PredictionData,
        plan_s_goal_m : Optional[float] = None,
        all_conditions: Dict[int, Tuple] = {},
        s_bounds: Optional[Tuple[float, float]] = None,
    ) -> Dict[int, Dict[str, float]]:
        """
        For each actor, find its s_min, s_max, l_min, l_max at a specific target time
        (or over a short horizon if sweeping).
        """
        out: Dict[int, Dict[str, float]] = {}

        max_route_len = planner_state.route_len
        route_index = planner_state.route_index
        lookahead_dist = plan_s_goal_m if plan_s_goal_m else self.sl_grid_spec.S_max
        lookahead_pts = self.config.meters_to_dense_route_idx(lookahead_dist)
        to_index = min(max_route_len, route_index + lookahead_pts + 1)
        route_slice = slice(route_index, to_index)

        route_pts = planner_state.original_route_points[route_slice]
        route_yaws = planner_state.original_rotation_angles[route_slice]
        s_route = planner_state.original_route_s[route_slice] - planner_state.original_route_s[route_index]

        # route_pts = planner_state.route_points[route_slice]
        # route_yaws = planner_state.rotation_angles[route_slice]
        # s_route = planner_state.s_route[route_slice] - planner_state.s_route[route_index]

        # Process obstacles
        ego_extent_x = self.config.ego_extent_x
        ego_extent_y = self.config.ego_extent_y

        all_obstacles = scene_data.obstacle_data.all_obstacles
        for obs_data in all_obstacles:
            obs_id = obs_data.id
            if obs_id in all_conditions:
                directive, actor_type, traffic_type, priority = all_conditions[obs_id]
            else:
                directive, actor_type, traffic_type, priority = "default", "default", "default", "medium"

            profile = resolve_dilation_profile(
                directive, actor_type, traffic_type, priority,
                ego_extent_x=ego_extent_x, ego_extent_y=ego_extent_y,
            )

            # Get obstacle bounding box XY corners
            obs_bb_corners_xy = obs_data.bbox_corners_xy

            # Get Frenet S and L bounds
            s_min, s_max, l_min, l_max = self.get_frenet_bounds(
                bbox_corners_xy_N=obs_bb_corners_xy,
                route_xy=route_pts[:, :2],
                route_s=s_route,
                route_yaws=route_yaws,
            )

            out[obs_id] = {
                "s_min": s_min,
                "s_max": s_max,
                "l_min": l_min,
                "l_max": l_max,
                "action_type": directive,
                "cost_scale": profile.cost_scale,
                "s_pad_before": profile.ds_sl_before,
                "s_pad_after": profile.ds_sl_after,
                "l_pad": profile.dl_sl,
                "decay_exponent": profile.decay_sl,
            }

        return out

    def find_grid_boundaries(
        self,
        planner_state: PlannerState,
        scene_data: SceneData,
        ego_plan: EgoPlan,
    ) -> Tuple[float, float, float]:
        route_index = planner_state.route_index
        cur_wp = planner_state.route_waypoints[route_index]
        half_w = cur_wp.lane_width / 2.0
        buf    = self.sl_grid_spec.lane_width_buffer_m

        # Lateral boundaries
        L_max =  half_w + buf
        L_min = -(half_w + buf)

        action = ego_plan.action.value
        if "left" in action:
            L_max += cur_wp.get_left_lane().lane_width
        elif "right" in action:
            L_min -= cur_wp.get_right_lane().lane_width
        else:
            lane_info = scene_data.route_data.lane_info
            if lane_info.has_left_lane:
                L_max += lane_info.left_wp.lane_width
            if lane_info.has_right_lane:
                L_min -= lane_info.right_wp.lane_width

        L_min = max(self.sl_grid_spec.L_min, L_min)
        L_max = min(self.sl_grid_spec.L_max, L_max)

        # Longitudinal boundaries
        # TODO: LLM GUIDED S_MAX DETERMINATION
        S_max = self.config.sl_grid_spec.S_max
        S_max = min(self.sl_grid_spec.S_max, S_max)

        return L_min, L_max, S_max

    def apply_road_cost(
        self,
        planner_state: PlannerState,
        scene_data: SceneData,
        ego_plan: EgoPlan,
        L_dyn: np.ndarray,
        S_dyn: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Build a 2-D road-alignment cost surface over the dynamic S-L grid.

        Uses an exponential cost function based on the lane's half-width to
        create a sharp well. The cost stays low near the centerline but spikes
        aggressively near the lane boundaries, capping safely at the max costs.
        """
        lane_info = scene_data.route_data.lane_info
        action    = ego_plan.action
        shift_left  = "left"  in action.value
        shift_right = "right" in action.value

        is_lane_change = action in {Action.CHANGE_LANE_LEFT,  Action.CHANGE_LANE_RIGHT}
        is_overtake    = action in {Action.OVERTAKE_LEFT,     Action.OVERTAKE_RIGHT}

        # ---- Select the reference lateral position ----
        cur_wp = planner_state.route_waypoints[planner_state.route_index]
        cur_lane_width = cur_wp.lane_width
        target_lane_width = cur_lane_width

        l_ref = 0.0
        l_norm_width = cur_lane_width

        if is_lane_change:
            if shift_left and lane_info.has_left_lane:
                target_lane_width = lane_info.left_wp.lane_width
                l_ref = 0.5 * cur_lane_width + 0.5 * target_lane_width
                l_norm_width = l_ref
            elif shift_right and lane_info.has_right_lane:
                target_lane_width = lane_info.right_wp.lane_width
                l_ref = -(0.5 * cur_lane_width + 0.5 * target_lane_width)
                l_norm_width = abs(l_ref)

        elif is_overtake:
            # if lane_info.same_direction_lane_change_available:
            #     if shift_left and lane_info.has_left_lane:
            #         l_ref =  lane_info.left_wp.lane_width
            #         l_norm_width = l_ref
            #     elif shift_right and lane_info.has_right_lane:
            #         l_ref = -lane_info.right_wp.lane_width
            #         l_norm_width = abs(l_ref)
            if shift_left and lane_info.has_left_lane:
                target_lane_width = lane_info.left_wp.lane_width
                l_ref = 0.5 * cur_lane_width + 0.5 * target_lane_width
                l_norm_width = l_ref
            elif shift_right and lane_info.has_right_lane:
                target_lane_width = lane_info.right_wp.lane_width
                l_ref = -(0.5 * cur_lane_width + 0.5 * target_lane_width)
                l_norm_width = abs(l_ref)

        # Retrieve explicit cost limits
        source_cost = getattr(self.sl_grid_spec, 'source_lane_cost', 50.0)
        target_cost = getattr(self.sl_grid_spec, 'target_lane_cost', 50.0)

        # Exponential tuning parameters
        half_width = l_norm_width / 2.0
        sharpness = 0.5  # Higher = flatter in the middle, steeper at the edges
        exp_denom = np.exp(sharpness) - 1.0

        # ---- Calculate 1D Lateral Cost ----

        # 1. Target Lane Sink
        # Clip ratio at 1.0 so the cost naturally plateaus at exactly target_cost outside the lane
        target_dev_ratio = np.clip(np.abs(L_dyn - l_ref) / half_width, 0.0, 1.0)
        l_cost_target = target_cost * (np.exp(sharpness * target_dev_ratio) - 1.0) / exp_denom

        if is_overtake and l_ref != 0.0:
            # 2. Origin Lane Sink (Double-Well)
            origin_dev_ratio = np.clip(np.abs(L_dyn - 0.0) / half_width, 0.0, 1.0)
            l_cost_origin = source_cost * (np.exp(sharpness * origin_dev_ratio) - 1.0) / exp_denom

            # W-shaped barrier
            l_cost_1d = np.minimum(l_cost_origin, l_cost_target)
            # l_cost_1d = l_cost_target
        else:
            # Standard single sink
            l_cost_1d = l_cost_target

        # ---- Broadcast uniformly across S: shape (S_len, L_len) ----
        cost_road = np.broadcast_to(
            l_cost_1d[np.newaxis, :],
            (len(S_dyn), len(L_dyn)),
        ).copy().astype(np.float32)

        # For overtakes, revert reference back to current lane for downstream logic
        if is_overtake:
            l_ref = 0.0

        return l_ref, target_lane_width, cost_road

    def build_costmap_from_bands(
        self,
        s: np.ndarray,
        l: np.ndarray,
        bands_by_id: Dict[int, Dict[str, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rasterise metric S-L actor bands onto a dynamic grid.

        Each actor contributes:
          • A binary core footprint written to the occupancy map.
          • A decaying cost envelope written to the cost map via max-composition.

        Bands whose envelopes fall entirely outside the dynamic grid window are
        silently skipped so that only scene-relevant actors incur compute.

        Args:
            s:            Dynamic longitudinal grid, shape (S_len,).  Starts at 0.
            l:            Dynamic lateral grid, shape (L_len,).  Subset of global L.
            bands_by_id:  Per-actor metric band descriptors from compute_actor_sl_bands.
            collision_cost: Base cost assigned to the actor core footprint.

        Returns:
            occupancy_map: uint8 array, shape (S_len, L_len).  1 = occupied core.
            cost_map:      float32 array, shape (S_len, L_len).  Blended cost field.
        """
        S_len = len(s)
        L_len = len(l)

        collision_cost = self.sl_grid_spec.collision_cost

        cost = np.zeros((S_len, L_len), dtype=np.float32)
        occ  = np.zeros((S_len, L_len), dtype=np.uint8)

        if not bands_by_id:
            return occ, cost

        # Resolution is shared with the global grid
        ds = self.sl_grid_spec.ds
        dl = self.sl_grid_spec.dl

        # Dynamic grid origins in metric space
        s_origin = s[0]
        l_origin = l[0]
        s_end    = s[-1]
        l_end    = l[-1]

        for actor_id, band in bands_by_id.items():
            s_min = band["s_min"]
            s_max = band["s_max"]
            l_min = band["l_min"]
            l_max = band["l_max"]

            s_pad_before = band["s_pad_before"]
            s_pad_after  = band["s_pad_after"]
            l_pad        = band["l_pad"]

            # ----------------------------------------------------------------
            # Trim pass: discard actors whose full cost envelope lies outside
            # the dynamic grid so they don't contribute stale grid writes.
            # ----------------------------------------------------------------
            if (s_max + s_pad_after) < s_origin or (s_min - s_pad_before) > s_end:
                continue
            if (l_max + l_pad) < l_origin or (l_min - l_pad) > l_end:
                continue

            decay_exp = band["decay_exponent"]
            max_cost = collision_cost * band["cost_scale"]

            # ----------------------------------------------------------------
            # Core grid indices — relative to the dynamic grid origin
            # ----------------------------------------------------------------
            i_s_core_start = np.clip(int((s_min - s_origin) / ds), 0, S_len - 1)
            i_s_core_end   = np.clip(int((s_max - s_origin) / ds), 0, S_len - 1)
            i_l_core_start = np.clip(int((l_min - l_origin) / dl), 0, L_len - 1)
            i_l_core_end   = np.clip(int((l_max - l_origin) / dl), 0, L_len - 1)

            # ----------------------------------------------------------------
            # Extended envelope indices (clipped to dynamic grid)
            # ----------------------------------------------------------------
            i_s_ext_start = np.clip(int((s_min - s_pad_before - s_origin) / ds), 0, S_len - 1)
            i_s_ext_end   = np.clip(int((s_max + s_pad_after  - s_origin) / ds), 0, S_len - 1)
            i_l_ext_start = np.clip(int((l_min - l_pad        - l_origin) / dl), 0, L_len - 1)
            i_l_ext_end   = np.clip(int((l_max + l_pad        - l_origin) / dl), 0, L_len - 1)

            # ----------------------------------------------------------------
            # Occupancy: mark core footprint
            # ----------------------------------------------------------------
            occ[i_s_core_start : i_s_core_end + 1,
                i_l_core_start : i_l_core_end + 1] = 1

            # ----------------------------------------------------------------
            # Cost envelope
            # ----------------------------------------------------------------
            ENV_S = i_s_ext_end - i_s_ext_start + 1
            ENV_L = i_l_ext_end - i_l_ext_start + 1

            if ENV_S <= 0 or ENV_L <= 0:
                continue

            # Grid-index coordinates for every cell in the envelope
            s_idx_grid = np.arange(i_s_ext_start, i_s_ext_end + 1)[:, None]  # (ENV_S, 1)
            l_idx_grid = np.arange(i_l_ext_start, i_l_ext_end + 1)[None, :]  # (1, ENV_L)

            # ---- Distance from each envelope cell to the nearest core cell ----
            # (in grid-index units, then converted to metres)

            # Longitudinal distance
            dist_s = np.zeros_like(s_idx_grid, dtype=np.float32)
            dist_s = np.where(s_idx_grid < i_s_core_start, i_s_core_start - s_idx_grid, dist_s)
            dist_s = np.where(s_idx_grid > i_s_core_end, s_idx_grid - i_s_core_end, dist_s)

            # Lateral distance
            dist_l = np.zeros_like(l_idx_grid, dtype=np.float32)
            dist_l = np.where(l_idx_grid < i_l_core_start, i_l_core_start - l_idx_grid, dist_l)
            dist_l = np.where(l_idx_grid > i_l_core_end, l_idx_grid - i_l_core_end, dist_l)

            # Convert distances back to meters
            dist_s_m = dist_s * ds
            dist_l_m = dist_l * dl

            # ---- Normalise by their respective pad radii ----
            # Front and rear longitudinal pads may differ (e.g. yield_for command).
            norm_s = np.zeros_like(dist_s_m, dtype=np.float32)
            norm_s = np.where(s_idx_grid < i_s_core_start, dist_s_m / max(s_pad_before, 1e-3), norm_s)
            norm_s = np.where(s_idx_grid > i_s_core_end, dist_s_m / max(s_pad_after, 1e-3), norm_s)

            norm_l = dist_l_m / max(l_pad, 1e-3)

            # L2 norm
            norm_total = np.clip(np.hypot(norm_s, norm_l), 0.0, 1.0)

            # ---- Apply decay and write via max-composition ----
            env_cost = (max_cost * (1.0 - norm_total) ** decay_exp).astype(np.float32)

            row_slice = slice(i_s_ext_start, i_s_ext_end + 1)
            col_slice = slice(i_l_ext_start, i_l_ext_end + 1)
            cost[row_slice, col_slice] = np.maximum(cost[row_slice, col_slice], env_cost)

        return occ, cost

    def plot_sl_map(
        self,
        costmap : np.ndarray,
        path : Optional[List[Tuple[int, int, float, float, float]]] = None
    ):
        l, s = self.L_arr, self.S_arr
        Sgrid, Lgrid = np.meshgrid(s, l, indexing="ij")  # (S,L)
        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.pcolormesh(Sgrid, Lgrid, costmap.astype(float),
                        shading="nearest", cmap="inferno", vmin=0.0, vmax=max(1.0, float(costmap.max())))
        fig.colorbar(im, ax=ax, label="cost")

        # Overlay the Dijkstra path
        if path is not None:
            s_vals = [entry[0] for entry in path]
            l_vals = [entry[1] for entry in path]

            ax.plot(
                s_vals,
                l_vals,
                lw=2.5,
                color="tab:blue",
                zorder=7,
                label="planned path",
            )
            ax.scatter(
                s_vals,
                l_vals,
                s=18,
                color="tab:blue",
                zorder=8,
            )


        ax.set_xlabel("arc length s (m)")
        ax.set_ylabel("lateral distance l (m)")
        ax.set_ylim(self.sl_grid_spec.L_min, self.sl_grid_spec.L_max + 1e-3)
        ax.set_title("S-L costmap")
        ax.legend(loc="best", frameon=True)
        ax.grid(True, lw=0.6, alpha=0.5)

        return fig, ax