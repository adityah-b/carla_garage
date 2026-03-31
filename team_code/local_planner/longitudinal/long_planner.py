import time
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from config import GlobalConfig

from privileged_route_planner import PlannerState
from team_code.scene_descriptor.scene_descriptor import SceneData
from team_code.actor_prediction.motion_prediction import PredictionData

from actor_prediction.collision_checker import CollisionInterval, LaneOverlapInterval

from .config_specs import *
from .planner_algo import PlannerAlgo
from .st_occupancy import STOccupancyGrid, STMaps


@dataclass
class LongPlannerResult:
    """Mirrors LatPlannerResult — bundles all longitudinal plan state in one object."""

    # Route anchor for the plan
    plan_route_start_idx: int = -1
    s_distance: float = 0.0        # planned horizon in metres

    # Time-indexed profiles (all same length after interpolation)
    vel_profile:  np.ndarray = field(default_factory=lambda: np.array([]))
    time_profile: np.ndarray = field(default_factory=lambda: np.array([]))
    s_profile:    np.ndarray = field(default_factory=lambda: np.array([]))
    plan_indices: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))

    # Execution cursor
    profile_step_idx: int = 0
    ego_idx:          int = 0

    # Plan metadata
    plan_cost:   float = float("inf")
    is_new_plan: bool  = False

    # ST maps — stored for logging/visualization
    st_maps: Optional[object] = None   # STMaps instance

    # ── derived properties ────────────────────────────────────────────────────

    @property
    def is_empty_plan(self) -> bool:
        return self.vel_profile.size == 0

    @property
    def has_active_plan(self) -> bool:
        return (self.vel_profile.size > 0
                and self.profile_step_idx < len(self.vel_profile))

    @property
    def target_speed(self) -> float:
        """Current commanded speed (m/s), or 0 when no active plan."""
        if not self.has_active_plan:
            return 0.0
        return float(self.vel_profile[self.profile_step_idx])

    def clear(self) -> None:
        self.plan_route_start_idx = -1
        self.s_distance           = 0.0
        self.vel_profile          = np.array([])
        self.time_profile         = np.array([])
        self.s_profile            = np.array([])
        self.plan_indices         = np.empty((0, 2), dtype=int)
        self.profile_step_idx     = 0
        self.ego_idx              = 0
        self.plan_cost            = float("inf")
        self.is_new_plan          = False
        self.st_maps              = None


class LongPlanner:
    def __init__(
        self,
        config : GlobalConfig,
        algo_name : str = 'dijkstra',
    ):
        self.config = config

        self.st_grid_spec : STGridSpec = self.config.st_grid_spec
        self.st_algo_spec : STAlgoSpec = self.config.st_algo_spec

        sim_freq = self.config.fps
        plan_freq = self.config.long_planning_frequency

        self.dt_sim = 1 / sim_freq
        self.sim_ticks_per_plan = max(1, int(round(sim_freq / plan_freq)))

        self.grid_mapper = STOccupancyGrid(
            config=self.config,
            st_grid_spec=self.st_grid_spec
        )
        self.planner = PlannerAlgo(
            config=self.config,
            st_algo_spec=self.st_algo_spec,
            algo_name=algo_name,
        )

        self.current_plan: LongPlannerResult = LongPlannerResult()

    # ── backward-compatible aliases (used by trajectory_planner.py) ───────────

    @property
    def has_active_plan(self) -> bool:
        return self.current_plan.has_active_plan

    @property
    def current_vel_profile(self) -> np.ndarray:
        return self.current_plan.vel_profile

    @property
    def profile_step_idx(self) -> int:
        return self.current_plan.profile_step_idx

    @property
    def plan_route_start_idx(self) -> int:
        return self.current_plan.plan_route_start_idx

    @property
    def current_st_maps(self) -> Optional[STMaps]:
        return self.current_plan.st_maps

    def reset_plan(self) -> None:
        self.current_plan.clear()

    def remaining_plan_samples(self) -> np.ndarray:
        """Return remaining (s, t) samples of the current plan from now onward."""
        plan = self.current_plan
        if not plan.has_active_plan:
            return np.empty((0, 2), dtype=float)

        if plan.profile_step_idx >= len(plan.time_profile):
            return np.empty((0, 2), dtype=float)

        current_time = float(plan.time_profile[plan.profile_step_idx])
        current_s    = float(plan.s_profile[plan.profile_step_idx])

        future_times = plan.time_profile[plan.profile_step_idx:].copy()
        future_s     = plan.s_profile[plan.profile_step_idx:].copy()

        future_times = np.maximum(future_times - current_time, 0.0)
        future_s     = np.maximum(future_s     - current_s,    0.0)

        return np.column_stack((future_s, future_times))

    def compute_new_plan(
        self,
        st_maps: STMaps,
        ego_speed: float,
        ego_max_speed: float,
        ego_accel : float,
        plan_route_start_idx: int,
        plan_s_goal_m: Optional[float] = None,
        *,
        force_replace: bool = False,
        ego_cruise_speed : Optional[float] = None,
        ego_goal_speed : Optional[float] = None,
    ) -> None:
        """Plan a new profile and (optionally) replace existing one if better/required."""
        start_idx = 0
        goal_idx = self._goal_idx_from_distance(plan_s_goal_m)
        print(f'\n\nLONG PLANNNER')
        print(f'\tPLAN_S: {plan_s_goal_m}, GOAL_IDX: {goal_idx}')
        print(f'\tACCEL: {ego_accel}')

        path, total_cost = self.planner.run(
            st_maps.occupancy_map,
            st_maps.cost_map,
            start_idx,
            goal_idx,
            v0=ego_speed,
            v_max=ego_max_speed,
            v_cruise_ref=ego_cruise_speed,
            v_goal_ref=ego_goal_speed,
            a0=ego_accel,
        )

        # No path found
        if len(path) <= 0:
            # If we *must* replace (no plan or unsafe), clear the current plan
            if force_replace or not self.has_active_plan:
                self.reset_plan()
            # Otherwise: keep the existing (safe) plan
            return

        # If we already have a plan and are not forced to replace,
        # only adopt the new plan if it has lower total cost.
        # --- thresholds ---
        TIME_IMPROVEMENT_FRAC = 0.10   # 10% faster
        COST_IMPROVEMENT_FRAC = 0.20   # 10% lower cost
        MAX_COST_INCREASE_IF_FASTER = 0.10  # allow up to +10% cost if >=10% faster

        # Decide whether to replace the current plan
        profiles_new = self.extract_profiles_from_path(
            path,
            self.grid_mapper.T_arr,
            self.grid_mapper.S_arr,
            self.st_algo_spec.dt_algo,
        )

        if not force_replace and self.has_active_plan and np.isfinite(self.current_plan_cost):
            old_cost = float(self.current_plan_cost)
            new_cost = float(total_cost)

            new_time_prof = profiles_new["t"]
            new_duration = float(new_time_prof[-1] - new_time_prof[0])

            if self.current_time_profile is not None and len(self.current_time_profile) > 0:
                old_duration = float(self.current_time_profile[-1] - self.current_time_profile[0])
            else:
                old_duration = float("inf")

            # Protect against degenerate old_duration
            if not np.isfinite(old_duration) or old_duration <= 1e-6:
                old_duration = float("inf")

            # Time priority rule:
            # 1) Replace if >=10% faster AND not more than +10% cost worse.
            faster_10pct = (new_duration <= (1.0 - TIME_IMPROVEMENT_FRAC) * old_duration)
            cost_not_too_much_worse = (new_cost <= (1.0 + MAX_COST_INCREASE_IF_FASTER) * old_cost)

            if faster_10pct and cost_not_too_much_worse:
                pass  # keep going -> replace
            else:
                # 2) Otherwise replace if >=10% lower cost.
                cheaper_10pct = (new_cost <= (1.0 - COST_IMPROVEMENT_FRAC) * old_cost)
                if not cheaper_10pct:
                    return
        # At this point we either:
        # - had no plan, or
        # - were forced to replace (unsafe), or
        # - found a cheaper plan.
        profiles = profiles_new

        time_prof = profiles["t"]
        vel_prof  = profiles["v"]
        s_prof    = profiles["s"]
        indices   = profiles["idx"]

        t0    = time_prof[0]
        t1    = time_prof[-1]
        t_sim = np.arange(t0, t1 + self.dt_sim, self.dt_sim)

        vel_interp = np.interp(t_sim, time_prof, vel_prof)
        s_interp   = np.interp(t_sim, time_prof, s_prof)

        # Commit to current_plan — keeps all state in one place
        self.current_plan.plan_route_start_idx = plan_route_start_idx
        self.current_plan.plan_cost            = float(total_cost)
        self.current_plan.vel_profile          = vel_interp[1:]
        self.current_plan.time_profile         = t_sim[1:]
        self.current_plan.s_profile            = s_interp[1:]
        self.current_plan.plan_indices         = indices[1:]
        self.current_plan.profile_step_idx     = 0
        self.current_plan.is_new_plan          = True

    def validate_plan_against_occupancy(
        self,
        occupancy_map: np.ndarray,
        plan_samples: np.ndarray,
    ) -> Tuple[bool, float]:
        """
        Validate an existing (s, t) plan against the current s–T occupancy grid.

        Args:
            occupancy_map: Boolean occupancy grid O(t_idx, s_idx).
            plan_samples: Array-like of shape (N, 2) with columns [s_j, t_j],
                where t0 = 0 for the start of the remaining plan.

        Returns:
            A tuple `(is_safe, ttc_min)` where:
                * is_safe: True if no sample lies in an occupied cell.
                * ttc_min: Earliest time-to-collision (seconds) or +inf if none.
        """
        if plan_samples is None or len(plan_samples) == 0:
            return True, float("inf")

        samples = np.asarray(plan_samples, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != 2:
            raise ValueError(
                f"plan_samples must have shape (N, 2) with [s, t]; got {samples.shape}"
            )

        dt = self.st_grid_spec.dt
        ds = self.st_grid_spec.ds

        t_collision = None

        for s_j, t_j in samples:
            i_t = int(np.clip(np.floor(t_j / dt), 0, occupancy_map.shape[0] - 1))
            i_s = int(np.clip(np.floor(s_j / ds), 0, occupancy_map.shape[1] - 1))

            # TODO: RENAME OCCUPANCY TO COST BASED CHECKS OR ACTUALLY PASS IN OCCUPANCY MAP
            # NOTE: TESTING VALIDITY CHECK (NOTE OCCUPANCY HERE IS CURRENTLY THE COSTMAP)
            if occupancy_map[i_t, i_s] > 1.0:
                print(f'FOUND COLLISION: (time: {t_j}, s: {s_j}), (time_idx: {i_t}, s_idx: {i_s})')
                t_collision = t_j
                break

        ttc_min = t_collision if t_collision is not None else float("inf")
        is_safe = t_collision is None

        return is_safe, ttc_min

    def run_step(
        self,
        plan_tick_counter : int,
        planner_state : PlannerState,
        scene_data : SceneData,
        prediction_data : PredictionData,
        all_conditions : Dict = {},
        plan_s_goal_m: Optional[float] = None,
        s_ego_m: float = 0.0,
        ego_cruise_speed : Optional[float] = None,
        ego_goal_speed : Optional[float] = None,
        s_bounds : Optional[Tuple[float, float]] = None,
        profile_time : bool = False,
    ) -> LongPlannerResult:
        """Run the longitudinal planner at simulation rate."""
        def bbox_to_vec7(bb) -> np.ndarray:
            return np.array([
                bb.location.x, bb.location.y, bb.location.z,
                bb.extent.x,   bb.extent.y,   bb.extent.z,
                bb.rotation.yaw
            ], dtype=np.float32)

        should_plan_now = (plan_tick_counter % self.sim_ticks_per_plan) == 1 or not self.has_active_plan

        ego_speed = scene_data.ego_data.speed
        max_speed = scene_data.traffic_data.speed_limit

        route_index = planner_state.route_index
        route_pts = planner_state.route_points[route_index:]

        self.current_plan.is_new_plan = False

        if should_plan_now:
            t_occ_start = time.perf_counter()
            st_maps = self.grid_mapper.build_st_occupancy(
                planner_state=planner_state,
                scene_data=scene_data,
                prediction_data=prediction_data,
                all_conditions=all_conditions,
                s_bounds=s_bounds,
            )
            t_occ_end = time.perf_counter()
            self.current_plan.st_maps = st_maps

            # TODO VERY EXPERIMENTAL
            # self.compute_new_plan(
            #     occupancy_map=occupancy_map,
            #     ego_speed=ego_speed,
            #     ego_max_speed=ego_max_speed,
            #     ego_accel=scene_data.ego_data.accel,
            #     plan_s_goal_m=plan_s_goal_m,
            #     plan_route_start_idx=route_index,
            #     force_replace=False,
            #     ego_cruise_speed=ego_cruise_speed,
            #     ego_goal_speed=ego_goal_speed,
            # )

            plan_safe = True
            if self.has_active_plan:
                all_actor_collisions = prediction_data.all_actor_collisions

                route_geometry = self.grid_mapper.precompute_route_geometry(route_pts)
                print(f'Velocity profile exists, checking safety')
                remaining_samples = self.remaining_plan_samples()

                # Rollout ego trajectory and check for collisions
                for actor_id, collisions in all_actor_collisions.items():
                    if not collisions:
                        continue

                    # TODO: FIGURE OUT A WAY TO HANDLE COLLISION CHECKS WHEN FIRST GENERATING A PLAN
                    # CAUSES PLANS TO FLICKER BACK AND FORTH BECAUSE EGO FORECAST PRIOR TO VEL PROFILE WITH TARGET SPEED
                    # PREDICTS COLLISIONS AND THEN WE CHECK THESE WITH THE NEW GENERATED PLAN WHICH HAS NO COLLISIONS
                    if self.profile_step_idx == 0:
                        continue
                    collision_interval = collisions[0]
                    start_idx = collision_interval.start_idx
                    ttc_min = start_idx * self.st_grid_spec.dt

                    # Check where the collision happens
                    ego_collision_bb = np.array([bbox_to_vec7(collision_interval.collision_bboxes_b[start_idx])])
                    corners_collision = self.grid_mapper.obb_corners_world_xy(ego_collision_bb)  # expected (T_collision, 4, 2)
                    ego_s_collision = self.grid_mapper.project_points_to_route_s(
                        corners_collision.reshape(-1, 2),
                        route_geometry,
                    ).reshape(1, 4)

                    # Reduce collision band to min and max s per timestep
                    s_min = np.min(ego_s_collision, axis=1)[0]
                    s_max = np.max(ego_s_collision, axis=1)[0]

                    if s_bounds is not None:
                        s_bound_min, s_bound_max = s_bounds
                        if (s_max < s_bound_min) or (s_min > s_bound_max):
                            print(f'Ignoring collision with actor {actor_id}, collision out of bounds')
                            continue
                        elif ttc_min > remaining_samples[-1, -1]:
                            print(f'Ignoring collision with actor: {actor_id}, ttc_min: {ttc_min}, plan_time_end: {remaining_samples[-1, -1]}')
                            continue

                    elif ttc_min > remaining_samples[-1, -1]:
                        print(f'Ignoring collision with actor: {actor_id}, ttc_min: {ttc_min}, plan_time_end: {remaining_samples[-1, -1]}')
                        continue

                    s_min_idx = np.clip(np.floor(s_min / self.st_grid_spec.ds), 0, st_maps.occupancy_map.shape[1] - 1).astype(np.int32)
                    s_max_idx = np.clip(np.floor(s_max / self.st_grid_spec.ds), 0, st_maps.occupancy_map.shape[1] - 1).astype(np.int32)
                    t_idx = int(np.clip(start_idx, 0, st_maps.occupancy_map.shape[0] - 1))

                    print(f'\n\nEGO ROLLOUT COLLISION')
                    print(f'\t\tactor_id: {actor_id}, s_min: {s_min}, s_max: {s_max}, s_min_idx: {s_min_idx}, s_max_idx: {s_max_idx}, t_idx: {t_idx}, ttc_min: {ttc_min}, plan_time_end: {remaining_samples[-1, -1]}')
                    # occupancy_map[t_idx, s_min_idx : s_max_idx] = 255.0
                    plan_safe = False
                    break

                # print(f'remaining samples shape: {remaining_samples.shape}')
                if plan_safe:
                    print(f'Safe plan, proceed')
                else:
                    print(f'Unsafe, replanning')
                    self.compute_new_plan(
                        st_maps=st_maps,
                        ego_speed=ego_speed,
                        ego_max_speed=max_speed,
                        ego_accel=scene_data.ego_data.accel,
                        plan_s_goal_m=plan_s_goal_m,
                        plan_route_start_idx=route_index,
                        force_replace=(not plan_safe),
                        ego_cruise_speed=ego_cruise_speed,
                        ego_goal_speed=ego_goal_speed,
                    )
            else:
                # OPTIONAL: if you ever want to *opportunistically* search for a cheaper
                # plan even when safe, you can call compute_new_plan here with
                # force_replace=False and let cost decide:
                #
                t_dijk_start = time.perf_counter()
                self.compute_new_plan(
                    st_maps=st_maps,
                    ego_speed=ego_speed,
                    ego_max_speed=max_speed,
                    ego_accel=scene_data.ego_data.accel,
                    plan_s_goal_m=plan_s_goal_m,
                    plan_route_start_idx=route_index,
                    force_replace=True,
                    ego_cruise_speed=ego_cruise_speed,
                    ego_goal_speed=ego_goal_speed,
                )
                t_dijk_end = time.perf_counter()

            # TODO: TESTING PREDICTION MODULE SYNC

            # route_geometry = self.grid_mapper.precompute_route_geometry(route_pts)
            # # Check where the collision happens
            # for idx, ego_bb in enumerate(prediction_data.ego_forecasted_bbs):
            #     ego_bb_arr = np.array([bbox_to_vec7(ego_bb)])
            #     corners_collision = self.grid_mapper.obb_corners_world_xy(ego_bb_arr)  # expected (T_collision, 4, 2)
            #     ego_s_collision = self.grid_mapper.project_points_to_route_s(
            #         corners_collision.reshape(-1, 2),
            #         route_geometry,
            #     ).reshape(1, 4)

            #     # Reduce collision band to min and max s per timestep
            #     s_min = np.min(ego_s_collision, axis=1)[0]
            #     s_max = np.max(ego_s_collision, axis=1)[0]

            #     s_min_idx = np.clip(np.floor(s_min / self.st_grid_spec.ds), 0, st_maps.occupancy_map.shape[1] - 1).astype(np.int32)
            #     s_max_idx = np.clip(np.floor(s_max / self.st_grid_spec.ds), 0, st_maps.occupancy_map.shape[1] - 1).astype(np.int32)
            #     t_idx = int(np.clip(idx, 0, st_maps.occupancy_map.shape[0] - 1))

            #     st_maps.occupancy_map[t_idx, s_min_idx : s_max_idx] = 1
            #     st_maps.cost_map[t_idx, s_min_idx : s_max_idx] = 155.0

            # TODO: TESTING PREDICTION MODULE SYNC

            # self.plot_st_map(self.grid_mapper, st_maps.cost_map, max_speed)
            # plt.show()
            # if self.has_active_plan:
            #     cur_t = self.current_time_profile[self.profile_step_idx]
            #     cur_v = self.current_vel_profile[self.profile_step_idx]
            #     self.plot_st_profiles(self.profiles, cur_t, ego_speed, cur_v)

        plan = self.current_plan

        if profile_time:
            print("--- LongPlanner runtime ---")
            print(f"ST costmap construction time: {(t_occ_end - t_occ_start)*1000:.2f} ms")
            print(f"ST Dijkstra planning time: {(t_dijk_end - t_dijk_start)*1000:.2f} ms")
        #     print(f"Corridor construction time: {(t_corr_end - t_corr_start)*1000:.2f} ms")
        #     print(f"SL QP planning time: {(t_qp_end - t_qp_start)*1000:.2f} ms")

        # TODO: CONFIGURABLE PREDICTION AND DISTANCE HORIZONS
        # TODO: PROPERLY IMPLEMENT THE EGO STATION CHECK, MAYBE PRECOMPUTE ROUTE DURING INITIALIZATION OR AFTER ROUTE CHANGES
        if plan.has_active_plan:
            print(f'Velocity profile exists, no 10Hz match yet')
            ego_idx = np.searchsorted(plan.s_profile, s_ego_m, side="left")
            print(f'ego_idx: {ego_idx}, profile step idx: {plan.profile_step_idx}, max_idx: {len(plan.s_profile) - 1}')

            if ego_idx > len(plan.s_profile) - 1:
                print(f'EGO IDX BEYOND PLAN HORIZON, REPLANNING')
                self.reset_plan()
                return self.current_plan

            # ego_idx = max(ego_idx, plan.profile_step_idx)
            plan.profile_step_idx += 1
            plan.ego_idx = ego_idx
        else:
            self.reset_plan()

        return self.current_plan

    def extract_profiles_from_path(
        self,
        path: List[Tuple[int, int, float, float]],
        t: np.ndarray,
        s: np.ndarray,
        dt_algo_res: float
    ) -> Dict[str, np.ndarray]:
        """
        Convert a Dijkstra path [(t_idx, s_idx, v, a), ...] into time-aligned profiles.

        Returns dict with:
        t: (N,) time
        s: (N,) arclength (m)
        v: (N,) velocity (m/s)
        a: (N,) acceleration (m/s^2)
        j: (N,) jerk (m/s^3), j[0]=0 by convention
        idx: (N, 2) integer indices into (t_idx, s_idx)
        """
        if not path:
            return {
                "t": np.array([]),
                "s": np.array([]),
                "v": np.array([]),
                "a": np.array([]),
                "j": np.array([]),
                "idx": np.empty((0, 2), dtype=int),
            }

        t_idx = np.array([p[0] for p in path], dtype=int)
        s_idx = np.array([p[1] for p in path], dtype=int)
        v     = np.array([p[2] for p in path], dtype=float)
        a     = np.array([p[3] for p in path], dtype=float)

        print(f'\tPATH S_IDX: {s_idx[-1]}')
        tt = t[t_idx]
        ss = s[s_idx]

        j = np.zeros_like(a)
        if a.size > 1:
            j[1:] = (a[1:] - a[:-1]) / float(dt_algo_res)

        return {
            "t": tt,
            "s": ss,
            "v": v,
            "a": a,
            "j": j,
            "idx": np.stack([t_idx, s_idx], axis=1),
        }

    def _goal_idx_from_distance(self, plan_s_goal_m: Optional[float]) -> int:
        """Convert a goal distance in meters to the closest valid s-index."""
        if plan_s_goal_m is None:
            return self.grid_mapper.S_len - 1

        ds = self.st_grid_spec.ds
        goal_idx = int(np.floor(plan_s_goal_m / ds))
        goal_idx = int(np.clip(goal_idx, 0, self.grid_mapper.S_len - 1))
        return goal_idx

    ########################################
    # Visualization methods for debugging
    ########################################

    def plot_st_map(
        self,
        st_grid_mapper : STOccupancyGrid,
        costmap : np.ndarray,
        envelope_slope : float,
        path: Optional[List[Tuple[int, int, float, float]]] = None,
        start_idx: Optional[int] = None,
        goal_idx: Optional[int] = None
    ):
        t, s = st_grid_mapper.T_arr, st_grid_mapper.S_arr
        Tgrid, Sgrid = np.meshgrid(t, s, indexing="ij")  # (K,S)
        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.pcolormesh(Tgrid, Sgrid, costmap.astype(float),
                        shading="nearest", cmap="inferno", vmin=0.0, vmax=max(1.0, float(costmap.max())))
        fig.colorbar(im, ax=ax, label="cost")

        # Envelope
        Tfin = float(t[-1])
        ax.plot([0.0, Tfin], [0.0, envelope_slope * Tfin],
                "--", lw=1.5, c="tab:red", label="envelope")

        # Optional: overlay start/goal guides on s-axis (t=0 / any t)
        if start_idx is not None:
            ax.scatter([t[0]], [s[start_idx]], s=60, c="tab:blue",
                    edgecolors="white", zorder=4, label="start")
        if goal_idx is not None:
            # show horizontal goal line for reference
            ax.axhline(s[goal_idx], color="tab:green", lw=1.2, alpha=0.6, label="goal s")

        # Overlay the Dijkstra path
        if self.has_active_plan:
            remaining_path = self.remaining_plan_samples()
            s_vals = remaining_path[:, 0]
            t_vals = remaining_path[:, 1]

            ax.plot(
                t_vals,
                s_vals,
                lw=2.5,
                color="tab:blue",
                zorder=7,
                label="planned path",
            )
            ax.scatter(
                t_vals,
                s_vals,
                s=18,
                color="tab:blue",
                zorder=8,
            )

        ax.set_xlabel("time t (s)")
        ax.set_ylabel("arc length s (m)")
        ax.set_ylim(0.0, self.st_grid_spec.S_max + 1e-3)
        ax.set_title("s–T costmap")
        ax.legend(loc="best", frameon=True)
        ax.grid(True, lw=0.6, alpha=0.5)

        return fig, ax

    def plot_st_profiles(
        self,
        profiles: Dict[str, np.ndarray],
        cur_time: float,
        ego_speed: float,
        target_speed: float,
        title: str = "Speed profile (s, v, a, j)",
    ):
        tt = profiles["t"]
        ss = profiles["s"]
        vv = profiles["v"]
        aa = profiles["a"]
        jj = profiles["j"]

        fig, axs = plt.subplots(4, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
        fig.suptitle(title)

        axs[0].plot(tt, ss, marker="o")
        axs[0].set_ylabel("s (m)")
        axs[0].grid(True, lw=0.6, alpha=0.5)

        axs[1].plot(tt, vv, marker="o", label="planned")
        axs[1].scatter([cur_time], [ego_speed], s=40, color="red", zorder=5, label="ego current")
        axs[1].scatter([cur_time], [target_speed], s=40, color="green", zorder=5, label="vel target")
        axs[1].set_ylabel("v (m/s)")
        axs[1].grid(True, lw=0.6, alpha=0.5)
        axs[1].legend(loc="best", frameon=True)

        axs[2].plot(tt, aa, marker="o")
        axs[2].set_ylabel("a (m/s²)")
        axs[2].grid(True, lw=0.6, alpha=0.5)

        axs[3].plot(tt, jj, marker="o")
        axs[3].set_ylabel("j (m/s³)")
        axs[3].set_xlabel("time t (s)")
        axs[3].grid(True, lw=0.6, alpha=0.5)

        for ax in axs:
            ax.axvline(cur_time, linestyle="--", linewidth=1.0, color="red", alpha=0.7)

        return fig, axs

# TODO: Add plotting/visualization methods for debugging