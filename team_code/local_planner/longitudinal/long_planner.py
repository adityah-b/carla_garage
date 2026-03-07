import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, Optional

from config import GlobalConfig
from actor_prediction.collision_checker import CollisionInterval, LaneOverlapInterval

from .config_specs import *
from .planner_algo import PlannerAlgo
from .st_occupancy import STOccupancyGrid

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
        self.planner = PlannerAlgo(algo_name=algo_name, st_algo_spec=self.st_algo_spec)

        # Planner state
        self.current_vel_profile: np.ndarray = np.array([])
        self.current_time_profile: np.ndarray = np.array([])
        self.current_s_profile: np.ndarray = np.array([])
        self.current_plan_indices: np.ndarray = np.empty((0, 2), dtype=int)
        self.profile_step_idx: int = 0
        self.ego_idx : int = 0
        self.cur_path = None
        self.current_plan_cost: float = float("inf")   # NEW
        # TODO: PROPERLY IMPLEMENT THE EGO STATION CHECK, MAYBE PRECOMPUTE ROUTE DURING INITIALIZATION OR AFTER ROUTE CHANGES
        self.plan_route_start_idx: int = -1

    @property
    def has_active_plan(self) -> bool:
        return self.current_vel_profile.size > 0 and self.profile_step_idx < len(self.current_vel_profile)

    def reset_plan(self) -> None:
        self.current_vel_profile = np.array([])
        self.current_time_profile = np.array([])
        self.current_s_profile = np.array([])
        self.current_plan_indices = np.empty((0, 2), dtype=int)
        self.profile_step_idx = 0
        self.current_plan_cost = float("inf")
        self.cur_path = None
        self.plan_route_start_idx = -1

    def remaining_plan_samples(self) -> np.ndarray:
        """Return remaining (s, t) samples of the current plan from now onward."""
        if not self.has_active_plan:
            return np.array([])

        if self.profile_step_idx >= len(self.current_time_profile):
            return np.array([])

        current_time = self.current_time_profile[self.profile_step_idx]
        future_times = self.current_time_profile[self.profile_step_idx :]
        future_times -= current_time
        future_times = np.maximum(future_times, 0.0)

        future_s = self.current_s_profile[self.profile_step_idx :]
        future_s -= self.current_s_profile[self.profile_step_idx]
        future_s = np.maximum(future_s, 0.0)

        # return np.stack([future_s, remaining_time], axis=1)
        return np.stack([future_s, future_times], axis=1)

    def compute_new_plan(
        self,
        occupancy_map: np.ndarray,
        ego_speed: float,
        ego_max_speed: float,
        plan_route_start_idx: int,
        plan_s_goal_m: Optional[float] = None,
        *,
        force_replace: bool = False,               # NEW
    ) -> None:
        """Plan a new profile and (optionally) replace existing one if better/required."""
        start_idx = 0
        goal_idx = self._goal_idx_from_distance(plan_s_goal_m)

        path, total_cost = self.planner.run(
            occupancy_map,
            start_idx,
            goal_idx,
            v0=ego_speed,
            v_max=ego_max_speed,
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
        self.cur_path = path
        print(f'new cost: {total_cost}, old cost: {self.current_plan_cost}')
        self.current_plan_cost = float(total_cost)

        # TODO: PROPERLY IMPLEMENT THE EGO STATION CHECK, MAYBE PRECOMPUTE ROUTE DURING INITIALIZATION OR AFTER ROUTE CHANGES
        self.plan_route_start_idx = plan_route_start_idx

        # profiles = self.extract_profiles_from_path(
        #     path,
        #     self.grid_mapper.T_arr,
        #     self.grid_mapper.S_arr,
        #     self.st_algo_spec.dt_algo,
        # )
        profiles = profiles_new
        self.profiles = profiles
        # self.plot_st_profiles(profiles)

        time_prof = profiles["t"]
        vel_prof = profiles["v"]
        s_prof   = profiles["s"]
        indices  = profiles["idx"]

        t0 = time_prof[0]
        t1 = time_prof[-1]
        t_sim = np.arange(t0, t1 + self.dt_sim, self.dt_sim)

        vel_interp = np.interp(t_sim, time_prof, vel_prof)
        s_interp = np.interp(t_sim, time_prof, s_prof)

        self.current_vel_profile = vel_interp
        self.current_time_profile = t_sim
        self.current_s_profile = s_interp
        self.current_plan_indices = indices
        self.profile_step_idx = 0

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

    # def run_step(
    #     self,
    #     ego_route_points_3d : np.ndarray,
    #     ego_speed : float,
    #     ego_max_speed : float,
    #     actor_collisions : Dict[int, List[CollisionInterval]], # K=actor id, V=collision intervals
    #     plan_tick_counter : int,
    #     plan_route_start_idx: int,
    #     all_conditions : Dict = {},
    #     plan_s_goal_m: Optional[float] = None,
    #     s_ego_m: float = 0.0,
    # ) -> np.ndarray:
    #     """Run the longitudinal planner at simulation rate."""

    #     should_plan_now = (plan_tick_counter % self.sim_ticks_per_plan) == 1 or not self.has_active_plan

    #     if should_plan_now:
    #         occupancy_map = self.grid_mapper.build_st_occupancy(
    #             route_points_3d=ego_route_points_3d,
    #             max_speed=ego_max_speed,
    #             actor_collisions=actor_collisions,
    #             all_conditions=all_conditions
    #         )

    #         plan_safe = False
    #         if self.has_active_plan:
    #             # NOTE: VERY EXPERIMENTAL, TRYING TO GENERATE A BETTER PLAN EVEN WHEN SAFE
    #             # self.compute_new_plan(
    #             #     occupancy_map=occupancy_map,
    #             #     ego_speed=ego_speed,
    #             #     ego_max_speed=ego_max_speed,
    #             #     plan_s_goal_m=plan_s_goal_m,
    #             #     plan_route_start_idx=plan_route_start_idx,
    #             #     force_replace=False,
    #             # )

    #             print(f'Velocity profile exists, checking safety')
    #             remaining_samples = self.remaining_plan_samples()
    #             is_safe, ttc_min = self.validate_plan_against_occupancy(
    #                 occupancy_map=occupancy_map,
    #                 plan_samples=remaining_samples,
    #             )

    #             print(f'remaining samples shape: {remaining_samples.shape}')
    #             print(f'ttc_min: {ttc_min}')
    #             # if ttc_min > 1.25 * remaining_samples[1][-1]:
    #             # if ttc_min > 1.25 * (2 * plan_s_goal_m / ego_max_speed):
    #             if is_safe:
    #                 plan_safe = True
    #                 print(f'Safe plan, proceed')
    #             else:
    #                 plan_safe = False
    #                 print(f'Unsafe, replanning')
    #                 self.compute_new_plan(
    #                     occupancy_map=occupancy_map,
    #                     ego_speed=ego_speed,
    #                     ego_max_speed=ego_max_speed,
    #                     plan_s_goal_m=plan_s_goal_m,
    #                     plan_route_start_idx=plan_route_start_idx,
    #                     force_replace=(not plan_safe),
    #                 )
    #         else:
    #             # OPTIONAL: if you ever want to *opportunistically* search for a cheaper
    #             # plan even when safe, you can call compute_new_plan here with
    #             # force_replace=False and let cost decide:
    #             #
    #             self.compute_new_plan(
    #                 occupancy_map=occupancy_map,
    #                 ego_speed=ego_speed,
    #                 ego_max_speed=ego_max_speed,
    #                 plan_s_goal_m=plan_s_goal_m,
    #                 plan_route_start_idx=plan_route_start_idx,
    #                 force_replace=False,
    #             )

    #         # self.plot_st_map(self.grid_mapper, occupancy_map, ego_max_speed, self.cur_path)

    #     vel_cmd = 0.0
    #     # TODO: CONFIGURABLE PREDICTION AND DISTANCE HORIZONS
    #     # TODO: PROPERLY IMPLEMENT THE EGO STATION CHECK, MAYBE PRECOMPUTE ROUTE DURING INITIALIZATION OR AFTER ROUTE CHANGES
    #     if self.has_active_plan:
    #         print(f'Velocity profile exists, no 10Hz match yet')
    #         ego_idx = np.searchsorted(self.current_s_profile, s_ego_m, side="left")
    #         print(f'ego_idx: {ego_idx}, profile step idx: {self.profile_step_idx}, max_idx: {len(self.current_s_profile) - 1}')

    #         if ego_idx > len(self.current_s_profile) - 1:
    #             print(f'EGO IDX BEYOND PLAN HORIZON, REPLANNING')
    #             vel_cmd = self.current_vel_profile[-1]
    #             self.reset_plan()
    #             return vel_cmd

    #         ego_idx = max(ego_idx, self.profile_step_idx)
    #         vel_cmd = self.current_vel_profile[ego_idx]

    #         self.profile_step_idx += 1
    #         self.profile_step_idx = max(ego_idx, self.profile_step_idx)

    #         # NOTE: TEMPORARY, CHECKING IF THIS FIXES FORECASTING IN TRAJECTORYPLANNER
    #         self.ego_idx = ego_idx

    #         # self.plot_st_profiles(self.profiles, self.current_time_profile[ego_idx], ego_speed, vel_cmd)
    #     # plt.show()

    #     return vel_cmd

    def run_step(
        self,
        ego_route_points_3d : np.ndarray,
        ego_speed : float,
        ego_max_speed : float,
        actor_collisions : Dict[int, List[CollisionInterval]], # K=actor id, V=collision intervals
        actor_overlaps : Dict[int, List[LaneOverlapInterval]], # K=actor id, V=overlap intervals
        plan_tick_counter : int,
        plan_route_start_idx: int,
        route_index : int,
        all_conditions : Dict = {},
        plan_s_goal_m: Optional[float] = None,
        s_ego_m: float = 0.0,
    ) -> np.ndarray:
        """Run the longitudinal planner at simulation rate."""
        def bbox_to_vec7(bb) -> np.ndarray:
            return np.array([
                bb.location.x, bb.location.y, bb.location.z,
                bb.extent.x,   bb.extent.y,   bb.extent.z,
                bb.rotation.yaw
            ], dtype=np.float32)

        should_plan_now = (plan_tick_counter % self.sim_ticks_per_plan) == 1 or not self.has_active_plan

        if should_plan_now:
            occupancy_map = self.grid_mapper.build_st_occupancy(
                route_points_3d=ego_route_points_3d,
                max_speed=ego_max_speed,
                actor_overlaps=actor_overlaps,
                all_conditions=all_conditions,
            )

            plan_safe = True
            if self.has_active_plan:
                # TODO VERY EXPERIMENTAL
                self.compute_new_plan(
                    occupancy_map=occupancy_map,
                    ego_speed=ego_speed,
                    ego_max_speed=ego_max_speed,
                    plan_s_goal_m=plan_s_goal_m,
                    plan_route_start_idx=route_index,
                    force_replace=False,
                )

                route_geometry = self.grid_mapper.precompute_route_geometry(ego_route_points_3d)
                print(f'Velocity profile exists, checking safety')
                remaining_samples = self.remaining_plan_samples()

                # Rollout ego trajectory and check for collisions
                for actor_id, collisions in actor_collisions.items():
                    if not collisions:
                        continue

                    collision_interval = collisions[0]
                    start_idx = collision_interval.start_idx
                    ttc_min = start_idx * self.st_grid_spec.dt

                    # Check when the collision happens
                    # if ttc_min > 6.0:
                    #     print(f'Ignoring collision with actor: {actor_id}, ttc_min: {ttc_min}, plan_time_end: {remaining_samples[-1, -1]}')
                    #     continue
                    if ttc_min > 1.0 + remaining_samples[-1, -1]:
                        print(f'Ignoring collision with actor: {actor_id}, ttc_min: {ttc_min}, plan_time_end: {remaining_samples[-1, -1]}')
                        continue

                    # Check where the collision happens
                    ego_collision_bb = np.array([bbox_to_vec7(collision_interval.collision_bboxes_a[start_idx])])
                    corners_collision = self.grid_mapper.obb_corners_world_xy(ego_collision_bb)  # expected (T_collision, 4, 2)
                    ego_s_collision = self.grid_mapper.project_points_to_route_s(
                        corners_collision.reshape(-1, 2),
                        route_geometry,
                    ).reshape(1, 4)

                    # Reduce collision band to min and max s per timestep
                    s_min = np.min(ego_s_collision, axis=1)[0]
                    s_max = np.max(ego_s_collision, axis=1)[0]

                    s_min_idx = np.clip(np.floor(s_min / self.st_grid_spec.ds), 0, occupancy_map.shape[1] - 1).astype(np.int32)
                    s_max_idx = np.clip(np.floor(s_max / self.st_grid_spec.ds), 0, occupancy_map.shape[1] - 1).astype(np.int32)
                    t_idx = int(np.clip(start_idx, 0, occupancy_map.shape[0] - 1))

                    print(f'\n\nEGO ROLLOUT COLLISION')
                    print(f'\t\tactor_id: {actor_id}, s_min: {s_min}, s_max: {s_max}, s_min_idx: {s_min_idx}, s_max_idx: {s_max_idx}, t_idx: {t_idx}, ttc_min: {ttc_min}, plan_time_end: {remaining_samples[-1, -1]}')
                    occupancy_map[t_idx, s_min_idx : s_max_idx] = 255.0
                    plan_safe = False
                    # break

                print(f'remaining samples shape: {remaining_samples.shape}')
                if plan_safe:
                    print(f'Safe plan, proceed')
                else:
                    print(f'Unsafe, replanning')
                    self.compute_new_plan(
                        occupancy_map=occupancy_map,
                        ego_speed=ego_speed,
                        ego_max_speed=ego_max_speed,
                        plan_s_goal_m=plan_s_goal_m,
                        plan_route_start_idx=route_index,
                        force_replace=(not plan_safe),
                    )
            else:
                # OPTIONAL: if you ever want to *opportunistically* search for a cheaper
                # plan even when safe, you can call compute_new_plan here with
                # force_replace=False and let cost decide:
                #
                self.compute_new_plan(
                    occupancy_map=occupancy_map,
                    ego_speed=ego_speed,
                    ego_max_speed=ego_max_speed,
                    plan_s_goal_m=plan_s_goal_m,
                    plan_route_start_idx=route_index,
                    force_replace=False,
                )

            # self.plot_st_map(self.grid_mapper, occupancy_map, ego_max_speed, self.cur_path)

        vel_cmd = 0.0
        # TODO: CONFIGURABLE PREDICTION AND DISTANCE HORIZONS
        # TODO: PROPERLY IMPLEMENT THE EGO STATION CHECK, MAYBE PRECOMPUTE ROUTE DURING INITIALIZATION OR AFTER ROUTE CHANGES
        if self.has_active_plan:
            print(f'Velocity profile exists, no 10Hz match yet')
            ego_idx = np.searchsorted(self.current_s_profile, s_ego_m, side="left")
            print(f'ego_idx: {ego_idx}, profile step idx: {self.profile_step_idx}, max_idx: {len(self.current_s_profile) - 1}')

            # if ego_idx > len(self.current_s_profile) - 1:
            #     print(f'EGO IDX BEYOND PLAN HORIZON, REPLANNING')
            #     vel_cmd = self.current_vel_profile[-1]
            #     self.reset_plan()
            #     return vel_cmd

            ego_idx = max(ego_idx, self.profile_step_idx)
            # vel_cmd = self.current_vel_profile[ego_idx]
            vel_cmd = self.current_vel_profile[self.profile_step_idx]

            self.profile_step_idx += 1
            # self.profile_step_idx = max(ego_idx, self.profile_step_idx)

            # NOTE: TEMPORARY, CHECKING IF THIS FIXES FORECASTING IN TRAJECTORYPLANNER
            self.ego_idx = ego_idx

            # self.plot_st_profiles(self.profiles, self.current_time_profile[self.profile_step_idx], ego_speed, vel_cmd)
        else:
            self.reset_plan()
        # plt.show()

        return vel_cmd

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

        # Optional: overlay the Dijkstra path
        if path and len(path) > 0:
            t_idx_path = np.array([p[0] for p in path], dtype=int)
            s_idx_path = np.array([p[1] for p in path], dtype=int)
            ax.plot(t[t_idx_path], s[s_idx_path],
                    lw=2.5, color="tab:blue", zorder=5, label="planned path")
            ax.scatter(t[t_idx_path], s[s_idx_path],
                    s=18, color="tab:blue", zorder=6)

        ax.set_xlabel("time t (s)")
        ax.set_ylabel("arc length s (m)")
        ax.set_ylim(0.0, self.st_grid_spec.S_max + 1e-3)
        ax.set_title("s–T costmap")
        ax.legend(loc="best", frameon=True)
        ax.grid(True, lw=0.6, alpha=0.5)
        return fig, ax

    def plot_st_profiles(
        self,
        profiles : Dict[str, np.ndarray],
        cur_time : float,
        ego_speed : float,
        target_speed : float,
        title: str = "Speed profile (s, v, a, j)"
    ):
        """
        Plot s(t), v(t), a(t), j(t) from a Dijkstra path and RETURN the profiles.

        Returns:
        fig, axs, profiles_dict (with keys 't','s','v','a','j')
        """
        tt, ss, vv, aa, jj = profiles["t"], profiles["s"], profiles["v"], profiles["a"], profiles["j"]

        fig, axs = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
        fig.suptitle(title)

        axs[0].plot(tt, ss, marker="o")
        axs[0].set_ylabel("s (m)")
        axs[0].grid(True, lw=0.6, alpha=0.5)

        axs[1].plot(tt, vv, marker="o")
        axs[1].scatter([cur_time], [ego_speed], s=40, color="red", zorder=5, label="ego current")
        axs[1].scatter([cur_time], [target_speed], s=40, color="green", zorder=5, label="vel target")
        axs[1].set_ylabel("v (m/s)")
        axs[1].grid(True, lw=0.6, alpha=0.5)

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