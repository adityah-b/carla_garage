import cv2
import carla
import time
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from config import GlobalConfig
from agents.navigation.local_planner import RoadOption

# Perception modules
from privileged_route_planner import PlannerState
from scene_descriptor.scene_descriptor import SceneData

# Prediction modules
from actor_prediction.motion_prediction import PredictionData
from actor_prediction.collision_checker import CollisionInterval
from actor_prediction.geometric_utils import GeometricUtils

# Behavioural planner modules
from team_code.scene_analyzer.parsers.ego_plan_pydantic_models import *

# Lateral planner modules
from team_code.local_planner.lateral.config_specs import *
from team_code.local_planner.lateral.sl_occupancy import SLOccupancyGrid, SLMaps
from team_code.local_planner.lateral.planner_algo import PlannerAlgo
from team_code.local_planner.lateral.sl_optimizer import SLSplineQPOptimizer

@dataclass
class LatPlannerResult:
    start_idx : int = -1
    goal_idx : int = -1

    s_distance : float = 0.0

    route_points : np.ndarray = field(default_factory=lambda : np.array([]))
    route_yaws : np.ndarray = field(default_factory=lambda : np.array([]))
    route_commands : np.ndarray = field(default_factory=lambda : np.array([]))

    is_new_plan : bool = False

    # SL plan data — populated on every replanning tick
    sl_maps  : Optional[object]    = None   # SLMaps (S_arr, L_arr, L_ref, cost_map)
    dp_path  : Optional[np.ndarray] = None  # (N, 2)  [s, l]  Dijkstra seed path
    corridor : Optional[np.ndarray] = None  # (N, 2)  [lb, ub] convex corridor bounds
    qp_path  : Optional[np.ndarray] = None  # (M, 2)  [s, l]  QP-optimised spline

    # Lane-change segment boundaries — local indices into the route slice
    # (i.e. relative to start_idx).  Populated whenever the QP path has a
    # meaningful lateral excursion (|L|_max >= 0.5 m), None otherwise.
    lc_out_end_local    : Optional[int] = None  # outbound LC complete, ego in target lane
    follow_end_local    : Optional[int] = None  # IDM-follow phase ends, return LC begins
    lc_return_end_local : Optional[int] = None  # return LC complete, ego back in source lane

    @property
    def is_empty_plan(self) -> bool:
        return self.route_points.size == 0

    def clear(self):
        self.start_idx = -1
        self.goal_idx = -1
        self.s_distance = 0.0
        self.route_points = np.array([])
        self.route_yaws = np.array([])
        self.route_commands = np.array([])
        self.is_new_plan = False
        self.sl_maps  = None
        self.dp_path  = None
        self.corridor = None
        self.qp_path  = None
        self.lc_out_end_local    = None
        self.follow_end_local    = None
        self.lc_return_end_local = None

class LatPlanner:
    def __init__(
        self,
        config : GlobalConfig,
        algo_name : str = 'sl_dijkstra',
    ):
        self.config = config

        self.sl_grid_spec : SLGridSpec = self.config.sl_grid_spec

        self.lat_grid_spec : LatGridSpec = self.config.lat_grid_spec
        self.lat_algo_spec : LatAlgoSpec = self.config.lat_algo_spec

        sim_freq = self.config.fps
        plan_freq_lo = self.config.lat_planning_frequency
        plan_freq_hi = self.config.lat_planning_frequency_high

        self.dt_sim = 1 / sim_freq
        self.sim_ticks_per_plan_lo = max(1, int(round(sim_freq / plan_freq_lo)))
        self.sim_ticks_per_plan_hi = max(1, int(round(sim_freq / plan_freq_hi)))

        self.grid_mapper = SLOccupancyGrid(config)
        self.planner_algo = PlannerAlgo(algo_name=algo_name, config=config)
        self.optimizer = SLSplineQPOptimizer(config)

        # Planner state
        self.current_plan = LatPlannerResult()

    @property
    def has_active_plan(self) -> bool:
        return not self.current_plan.is_empty_plan

    def reset_plan(self) -> None:
        self.current_plan.clear()

    def compute_ego_frenet_state(
        self,
        planner_state: PlannerState,
        scene_data: SceneData,
    ) -> Tuple[float, float, float]:
        """
        Compute the ego's initial Frenet state (l0, dl0/ds, d²l0/ds²) relative
        to the original route centerline at the current route index.

        Convention: L > 0 = left of route, L < 0 = right of route.

        Returns (l0, dl0, ddl0).
        """
        if scene_data is None or scene_data.ego_data is None:
            return 0.0, 0.0, 0.0

        route_index = planner_state.route_index
        ref_xy    = planner_state.original_route_points[route_index, :2]
        # rotation_angles stored in degrees; convert to radians for trig.
        ref_theta = np.deg2rad(planner_state.original_rotation_angles[route_index])

        ego_xy    = scene_data.ego_data.position   # [x, y] metres
        ego_theta = scene_data.ego_data.orientation # radians

        dx = ego_xy[0] - ref_xy[0]
        dy = ego_xy[1] - ref_xy[1]

        # Left normal: n = (sinθ, -cosθ) → positive L is to the left of the route.
        l0 = float(dx * np.sin(ref_theta) + dy * (-np.cos(ref_theta)))

        # Heading error normalised to [-π, π].
        heading_error = (ego_theta - ref_theta + np.pi) % (2 * np.pi) - np.pi

        # dL/dS ≈ tan(heading_error); clamp to ±45° to keep the QP well-posed.
        dl0  = float(np.tan(np.clip(heading_error, -np.pi / 4, np.pi / 4)))
        ddl0 = 0.0  # no curvature-rate estimate available

        return l0, dl0, ddl0

    def validate_plan_against_occupancy(
        self,
        occupancy_map: np.ndarray,
        path: np.ndarray,
    ) -> Tuple[bool, float]:
        pass

    def s_idx_from_distance(self, distance: float, S_max : float) -> int:
        """Convert a distance in meters to the closest valid s-index."""
        ds = self.sl_grid_spec.ds
        S_max_idx = int(np.floor(S_max / ds))

        s_idx = int(np.floor(distance / ds))
        s_idx = int(np.clip(s_idx, 0, S_max_idx))

        return s_idx

    def l_idx_from_distance(self, distance : float, L_min : float, L_max : float) -> int:
        """Convert a distance in meters to the closest valid l-index."""
        dl = self.sl_grid_spec.dl
        L_min_idx = 0.0
        L_max_idx = int(np.floor((L_max - L_min) / dl))

        l_idx = int(np.floor((distance - L_min)/ dl))
        l_idx = int(np.clip(l_idx, L_min_idx, L_max_idx))

        return l_idx

    def extract_path_corridor(
        self,
        sl_maps: SLMaps,
        dp_path: np.ndarray,
    ) -> np.ndarray:
        cost_map = sl_maps.cost_map
        s_arr    = sl_maps.S_arr
        l_arr    = sl_maps.L_arr

        dp_s_vals = dp_path[:, 0]
        dp_l_vals = dp_path[:, 1]

        # ── 1. Nearest-neighbour lookup (fix: outer subtract for correct broadcast) ──
        s_indices = np.argmin(np.abs(s_arr[:, None] - dp_s_vals[None, :]), axis=0)  # (N,)
        l_indices = np.argmin(np.abs(l_arr[:, None] - dp_l_vals[None, :]), axis=0)  # (N,)

        S, L  = cost_map.shape
        blocked   = cost_map >= 0.8 * self.sl_grid_spec.collision_cost          # (S, L)  bool mask
        idx_grid  = np.arange(L)              # (L,)

        # ── 2. Left barrier map ──────────────────────────────────────────────────────
        # left_barrier[s, l] = rightmost j ≤ l where cost[s,j] ≥ 50  (or 0 if none)
        left_idx_map = np.where(blocked, idx_grid[None, :], -1)           # (S, L)
        left_barrier = np.maximum.accumulate(left_idx_map, axis=1)        # (S, L)
        left_barrier = np.where(left_barrier < 0, 0, left_barrier)        # clamp to boundary

        # ── 3. Right barrier map ─────────────────────────────────────────────────────
        # right_barrier[s, l] = leftmost j ≥ l where cost[s,j] ≥ 50  (or L-1 if none)
        right_idx_map = np.where(blocked, idx_grid[None, :], L)           # (S, L)
        right_barrier = np.minimum.accumulate(
            right_idx_map[:, ::-1], axis=1)[:, ::-1]                      # (S, L)
        right_barrier = np.where(right_barrier >= L, L - 1, right_barrier)

        # ── 4. Gather bounds for each dp point ──────────────────────────────────────
        lb = l_arr[left_barrier [s_indices, l_indices]]   # (N,)
        ub = l_arr[right_barrier[s_indices, l_indices]]   # (N,)

        # ── 5. Degenerate-interval fix (vectorized) ──────────────────────────────────
        invalid = lb > ub
        mid     = (lb + ub) / 2.0
        lb      = np.where(invalid, mid - 0.01, lb)
        ub      = np.where(invalid, mid + 0.01, ub)

        return np.stack([lb, ub], axis=1)   # (N, 2)

    def extract_path_knots(
        self,
        planner_state : PlannerState,
        scene_data : SceneData,
        ego_plan : EgoPlan,
        s_path : np.ndarray,
    ) -> np.ndarray:
        # TODO: DYNAMICALLY CHOOSE KNOTS
        return s_path

    # TODO: DETERMINE IF LAT PLANNER NEEDS STATE MACHINE TO BE PASSED IN
    def run_step(
        self,
        plan_tick_counter : int,
        planner_state : PlannerState,
        lidar_data : Dict,
        scene_data : SceneData,
        prediction_data : PredictionData,
        ego_plan : EgoPlan,
        *,
        s_ego_m : float = 0.0,
        plan_s_goal_m : Optional[float] = None,
        all_conditions : Dict = {},
        s_bounds : Optional[Tuple[float, float]] = None,
        profile_time : bool = False,
    ) -> LatPlannerResult:
        self.current_plan.is_new_plan = False

        plan_lo = (plan_tick_counter % self.sim_ticks_per_plan_lo) == 1
        plan_hi = (plan_tick_counter % self.sim_ticks_per_plan_hi) == 1
        should_plan_now = plan_lo or plan_hi

        if should_plan_now:
            # Extract planner state data
            max_route_len = planner_state.route_len
            route_index = planner_state.route_index
            route_pts = planner_state.original_route_points
            route_yaws = planner_state.original_rotation_angles
            route_cmds = planner_state.route_commands
            s_route = planner_state.original_route_s
            s_route_max = s_route[-1] - s_route[route_index]

            plan_s_goal_m = plan_s_goal_m if plan_s_goal_m else min(s_route_max, self.sl_grid_spec.S_max)

            # Get occupancy and cost maps
            t_occ_start = time.perf_counter()
            sl_maps = self.grid_mapper.build_maps(
                planner_state=planner_state,
                scene_data=scene_data,
                prediction_data=prediction_data,
                ego_plan=ego_plan,
                all_conditions=all_conditions,
                s_bounds=s_bounds,
            )
            t_occ_end = time.perf_counter()

            # Prepare planner indices
            S_max = sl_maps.S_arr[-1]

            s_start_idx = 0
            s_goal_idx = self.s_idx_from_distance(plan_s_goal_m, S_max)

            L_min = sl_maps.L_arr[0]
            L_max = sl_maps.L_arr[-1]

            # Compute the ego's actual Frenet state relative to the original route.
            # ego_l0, ego_dl0, ego_ddl0 = self.compute_ego_frenet_state(planner_state, scene_data)
            ego_l0, ego_dl0, ego_ddl0 = 0.0, 0.0, 0.0

            l_start_idx = self.l_idx_from_distance(ego_l0, L_min=L_min, L_max=L_max)

            print(f'\n\nEGO INITIAL STATE')
            print(f'\tL0: {ego_l0}, DL0: {ego_dl0}, DDL0: {ego_ddl0}')

            # Get rough Dijkstra SL plan
            t_dijk_start = time.perf_counter()
            dp_path, cost = self.planner_algo.run(
                occupancy_map=sl_maps.occupancy_map,
                cost_map=sl_maps.cost_map,
                start_idx=s_start_idx,
                goal_idx=s_goal_idx,
                L_min=L_min,
                l_start_idx=l_start_idx,
                l_ref_m=sl_maps.L_ref,
            )
            t_dijk_end = time.perf_counter()

            if dp_path.size > 0:
                # Extract traversible corridor from DP path
                t_corr_start = time.perf_counter()
                corridor = self.extract_path_corridor(sl_maps=sl_maps, dp_path=dp_path)
                t_corr_end = time.perf_counter()

                # Extract path knots
                knots = self.extract_path_knots(
                    planner_state=planner_state,
                    scene_data=scene_data,
                    ego_plan=ego_plan,
                    s_path=dp_path[:, 0],
                )
                self.optimizer.set_knots(knots)

                init_state = (ego_l0, ego_dl0, ego_ddl0)
                # Solve spline QP optimization
                t_qp_start = time.perf_counter()
                qp_path = self.optimizer.solve(
                    dp_path=dp_path,
                    dp_bounds=corridor,
                    init_state=init_state,
                )
                t_qp_end = time.perf_counter()

                # self.plot_path_and_profiles(
                #     sl_maps=sl_maps,
                #     dp_path=dp_path,
                #     corridor=corridor,
                #     qp_path=qp_path
                # )

                if qp_path.size > 0:
                    # Revert from Frenet to Cartesian coordinates
                    lookahead_dist = plan_s_goal_m
                    lookahead_pts = self.config.meters_to_dense_route_idx(lookahead_dist)
                    to_index = min(max_route_len, route_index + lookahead_pts + 1)
                    route_slice = slice(route_index, to_index)

                    route_pts = route_pts[route_slice].copy()
                    route_yaws = route_yaws[route_slice].copy()
                    route_cmds = route_cmds[route_slice].copy()
                    s_route = s_route[route_slice] - s_route[route_index]

                    qp_s = qp_path[:, 0]
                    qp_l = qp_path[:, 1]

                    # 2. Interpolate the lateral shift (L) at the exact S-values of the route slice.
                    # This guarantees the new Frenet path has exactly the same length as route_pts.
                    # Convention: L > 0 = left shift (overtake left), L < 0 = right shift.
                    matched_l = np.interp(s_route, qp_s, qp_l)

                    # 3. Detect LC segment boundaries.
                    # matched_l is 1-to-1 with the route slice, so local index i maps to
                    # global route index (route_index + i).
                    lc_out_end_local    = None
                    follow_end_local    = None
                    lc_return_end_local = None

                    l_abs = np.abs(matched_l)
                    l_max = float(l_abs.max())

                    cur_wp = planner_state.original_route_waypoints[route_index]
                    L_src_edge = 0.5 * cur_wp.lane_width + self.config.lateral_buffer_m
                    L_ref_center = 0.5 * cur_wp.lane_width + 0.5 * sl_maps.target_lane_width

                    above = l_abs >= L_src_edge
                    if np.any(above):
                        lc_complete = l_abs >= L_ref_center - (L_src_edge / 2.0)
                        if np.any(lc_complete):
                            lc_out_end_local = int(np.argmax(lc_complete))
                        else:
                            lc_out_end_local = int(np.argmax(l_abs))

                        follow_end_local = int(np.argmax(l_abs))

                        # A return LC is present only when the end of the path has
                        # come back close to the source lane
                        back_return = l_abs[follow_end_local:] < L_src_edge
                        if np.any(back_return):
                            # lc_return_end_local = first index after follow_end where
                            # |L| drops below threshold (source lane restored).
                            return_lc_complete = l_abs[follow_end_local:] <= (L_src_edge / 2.0)
                            if np.any(return_lc_complete):
                                lc_return_end_local = follow_end_local + int(np.argmax(return_lc_complete))
                            else:
                                lc_return_end_local = len(matched_l) - 1

                        # Annotate route_commands. Sign of peak L determines direction.
                        l_peak  = matched_l[np.argmax(l_abs)]
                        out_cmd = RoadOption.CHANGELANELEFT  if l_peak > 0 else RoadOption.CHANGELANERIGHT

                        route_cmds[ : lc_out_end_local] = out_cmd
                        route_cmds[lc_out_end_local : follow_end_local] = RoadOption.LANEFOLLOW
                        if np.any(back_return):
                            ret_cmd = RoadOption.CHANGELANERIGHT if l_peak > 0 else RoadOption.CHANGELANELEFT
                            route_cmds[follow_end_local : lc_return_end_local] = ret_cmd

                    # 4. Stack into the matched (M, 2) shape expected by your converter
                    matched_frenet_path = np.column_stack((s_route, matched_l))

                    # 5. Convert back to Cartesian
                    route_xy, route_yaws = GeometricUtils.frenet_to_cartesian(
                        frenet_path=matched_frenet_path,
                        route_xy=route_pts[:, :2],
                        route_yaws=route_yaws,
                        s_route=s_route,
                    )
                    route_pts[:, :2] = route_xy

                    self.current_plan = LatPlannerResult(
                        start_idx=route_index,
                        goal_idx=to_index,
                        s_distance=plan_s_goal_m,
                        route_points=route_pts,
                        route_yaws=route_yaws,
                        route_commands=route_cmds,
                        is_new_plan=True,
                        sl_maps=sl_maps,
                        dp_path=dp_path,
                        corridor=corridor,
                        qp_path=qp_path,
                        lc_out_end_local=lc_out_end_local,
                        follow_end_local=follow_end_local,
                        lc_return_end_local=lc_return_end_local,
                    )
                else:
                    self.current_plan = LatPlannerResult()
            else:
                self.current_plan = LatPlannerResult()

        if profile_time:
            print("--- LatPlanner runtime ---")
            print(f"SL costmap construction time: {(t_occ_end - t_occ_start)*1000:.2f} ms")
            print(f"SL Dijkstra planning time: {(t_dijk_end - t_dijk_start)*1000:.2f} ms")
            print(f"Corridor construction time: {(t_corr_end - t_corr_start)*1000:.2f} ms")
            print(f"SL QP planning time: {(t_qp_end - t_qp_start)*1000:.2f} ms")

        return self.current_plan

    ########################################
    # Visualization methods for debugging
    ########################################

    def plot_path_and_profiles(
        self,
        sl_maps : SLMaps,
        dp_path : np.ndarray,
        corridor : np.ndarray,
        qp_path : np.ndarray,
    ):
        s_arr = sl_maps.S_arr
        l_arr = sl_maps.L_arr
        cost_map = sl_maps.cost_map

        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(4, 2, width_ratios=[1.2, 1])

        ax_map = fig.add_subplot(gs[:, 0])
        Sgrid, Lgrid = np.meshgrid(s_arr, l_arr, indexing="ij")
        im = ax_map.pcolormesh(Sgrid, Lgrid, cost_map, shading="nearest", cmap="inferno", vmin=0, vmax=255)
        fig.colorbar(im, ax=ax_map, label="Cost", fraction=0.046, pad=0.04)

        s_dp = dp_path[:, 0]
        l_dp = dp_path[:, 1]

        lb_vals = corridor[:, 0]
        ub_vals = corridor[:, 1]
        ax_map.fill_between(s_dp, lb_vals, ub_vals, color='white', alpha=0.15, label="Convex Corridor")
        ax_map.plot(s_dp, lb_vals, 'w--', lw=0.8)
        ax_map.plot(s_dp, ub_vals, 'w--', lw=0.8)

        ax_map.plot(s_dp, l_dp, color="cyan", lw=1.5, ls="--", marker='.', ms=6, label="Dijkstra Seed")

        if qp_path is not None:
            s_qp = qp_path[:, 0]
            l_qp = qp_path[:, 1]
            ax_map.plot(s_qp, l_qp, color="#00FF00", lw=3.0, zorder=10, label="Cubic Spline QP")

        ax_map.set_xlabel("Arc length S (m)")
        ax_map.set_ylabel("Lateral offset L (m)")
        ax_map.set_title("SL Costmap & Planned Paths")
        ax_map.legend(loc="upper left", frameon=True)
        ax_map.grid(True, lw=0.5, alpha=0.3, color='white')

        if qp_path is None:
            plt.show()
            return

        s = np.array(s_qp)
        l = np.array(l_qp)
        ds = s[1] - s[0]
        dl = np.gradient(l, ds)
        ddl = np.gradient(dl, ds)
        dddl = np.gradient(ddl, ds)

        axes = [fig.add_subplot(gs[i, 1]) for i in range(4)]

        axes[0].plot(s, l, color="#00FF00", lw=2)
        axes[0].set_ylabel("L (m)\nOffset")
        axes[0].set_title("Path Kinematic Profiles (Spline Model)")

        axes[1].plot(s, dl, color="dodgerblue", lw=2)
        axes[1].set_ylabel("L'\nHeading")

        axes[2].plot(s, ddl, color="orange", lw=2)
        axes[2].set_ylabel("L''\nCurvature")

        axes[3].plot(s, dddl, color="red", lw=2)
        axes[3].set_ylabel("L'''\nJerk")
        axes[3].set_xlabel("Arc length S (m)")

        for ax in axes:
            ax.grid(True, alpha=0.5, linestyle='--')
            ax.set_xlim(s[0], s[-1])
            if ax != axes[-1]:
                ax.set_xticklabels([])

        plt.tight_layout()
        plt.show()
