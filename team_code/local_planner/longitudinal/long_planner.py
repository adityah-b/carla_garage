import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, Optional

from actor_prediction.collision_checker import CollisionInterval

from .config_specs import *
from .planner_algo import PlannerAlgo
from .st_occupancy import STOccupancyGrid

class LongPlanner:
    def __init__(
        self,
        st_grid_spec : STGridSpec,
        st_algo_spec : STAlgoSpec,
        algo_name : str = 'dijkstra',
        sim_freq : float = 20.0,
        plan_freq : float = 10.0,
    ):
        self.st_grid_spec = st_grid_spec
        self.st_algo_spec = st_algo_spec

        self.dt_sim = 1 / sim_freq
        self.sim_ticks_per_plan = max(1, int(round(sim_freq / plan_freq)))

        self.grid_mapper = STOccupancyGrid(st_grid_spec=st_grid_spec)
        self.planner = PlannerAlgo(algo_name=algo_name, st_algo_spec=st_algo_spec)

        # Planner state
        self.current_vel_profile: np.ndarray = np.array([])
        self.current_time_profile: np.ndarray = np.array([])
        self.current_s_profile: np.ndarray = np.array([])
        self.current_plan_indices: np.ndarray = np.empty((0, 2), dtype=int)
        self.profile_step_idx: int = 0
        self.cur_path = None

    @property
    def has_active_plan(self) -> bool:
        return self.current_vel_profile.size > 0 and self.profile_step_idx < len(self.current_vel_profile)

    def remaining_plan_samples(self) -> np.ndarray:
        """Return remaining (s, t) samples of the current plan from now onward."""
        if not self.has_active_plan:
            return np.array([])

        if self.profile_step_idx >= len(self.current_time_profile):
            return np.array([])

        current_time = self.current_time_profile[self.profile_step_idx]
        future_times = self.current_time_profile[self.profile_step_idx :]
        future_s = self.current_s_profile[self.profile_step_idx :]

        remaining_time = future_times - current_time
        # return np.stack([future_s, remaining_time], axis=1)
        return np.stack([future_s, future_times], axis=1)

    def compute_new_plan(
        self,
        occupancy_map: np.ndarray,
        ego_speed: float,
        ego_max_speed: float,
    ) -> None:
        """Plan a new profile and reset state tracking."""
        start_idx = 0
        goal_idx = self.grid_mapper.S_len - 1
        path = self.planner.run(
            occupancy_map,
            start_idx,
            goal_idx,
            v0=ego_speed,
            v_max=ego_max_speed,
        )

        if len(path) <= 0:
            self.current_vel_profile = np.array([])
            self.current_time_profile = np.array([])
            self.current_s_profile = np.array([])
            self.current_plan_indices = np.empty((0, 2), dtype=int)
            self.profile_step_idx = 0
            return

        # NOTE: DEBUG
        self.cur_path = path

        profiles = self.extract_profiles_from_path(
            path,
            self.grid_mapper.T_arr,
            self.grid_mapper.S_arr,
            self.st_algo_spec.dt_algo_res,
        )

        time_prof = profiles["t"]
        vel_prof = profiles["v"]
        s_prof = profiles["s"]
        indices = profiles["idx"]

        t0 = time_prof[0]
        t1 = time_prof[-1]
        t_sim = np.arange(t0, t1 + 1e-9, self.dt_sim)

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
            if occupancy_map[i_t, i_s] > 0.5:
                t_collision = t_j
                break

        ttc_min = t_collision if t_collision is not None else float("inf")
        is_safe = t_collision is None

        return is_safe, ttc_min

    def run_step(
        self,
        ego_route_points_3d : np.ndarray,
        ego_speed : float,
        ego_max_speed : float,
        actor_collisions : Dict[int, List[CollisionInterval]], # K=actor id, V=collision intervals
        plan_tick_counter : int,
        all_conditions : Dict = {}
    ) -> np.ndarray:
        """Run the longitudinal planner at simulation rate.

        The planner runs collision checks and replanning at `plan_freq` (10 Hz by
        default). At intermediate simulation ticks (20 Hz by default) it advances
        along the stored velocity profile.
        """
        should_plan_now = (plan_tick_counter % self.sim_ticks_per_plan) == 1 or not self.has_active_plan

        if should_plan_now:
            occupancy_map = self.grid_mapper.build_st_occupancy(
                route_points_3d=ego_route_points_3d,
                max_speed=ego_max_speed,
                actor_collisions=actor_collisions,
                all_conditions=all_conditions
            )

            # self.plot_st_map(self.grid_mapper, occupancy_map, ego_max_speed, self.cur_path)
            # plt.show()

            plan_safe = False
            if self.has_active_plan:
                print(f'Velocity profile exists, checking safety')
                remaining_samples = self.remaining_plan_samples()
                _, ttc_min = self.validate_plan_against_occupancy(
                    occupancy_map=occupancy_map,
                    plan_samples=remaining_samples,
                )

                # TODO: SET ACTUAL TTC VALUE
                if ttc_min > 2.0:
                    plan_safe = True
                    print(f'Safe plan, proceed')
                else:
                    plan_safe = False
                    print(f'Unsafe, replanning')

            if (not self.has_active_plan) or (not plan_safe):
                print(f'Generating new plan')
                self.compute_new_plan(
                    occupancy_map=occupancy_map,
                    ego_speed=ego_speed,
                    ego_max_speed=ego_max_speed,
                )

        vel_cmd = 0.0
        if self.has_active_plan:
            print(f'Velocity profile exists, no 10Hz match yet')
            idx = min(self.profile_step_idx, len(self.current_vel_profile) - 1)
            vel_cmd = float(self.current_vel_profile[idx])
            self.profile_step_idx = min(
                self.profile_step_idx + 1, len(self.current_vel_profile)
            )

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
        ax.set_ylim(0.0, 40.0 + 1e-3)
        ax.set_title("s–T costmap")
        ax.legend(loc="best", frameon=True)
        ax.grid(True, lw=0.6, alpha=0.5)
        return fig, ax

    def plot_st_profiles(
        self,
        profiles : Dict[str, np.ndarray],
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
        axs[1].set_ylabel("v (m/s)")
        axs[1].grid(True, lw=0.6, alpha=0.5)

        axs[2].plot(tt, aa, marker="o")
        axs[2].set_ylabel("a (m/s²)")
        axs[2].grid(True, lw=0.6, alpha=0.5)

        axs[3].plot(tt, jj, marker="o")
        axs[3].set_ylabel("j (m/s³)")
        axs[3].set_xlabel("time t (s)")
        axs[3].grid(True, lw=0.6, alpha=0.5)

        return fig, axs

# TODO: Add plotting/visualization methods for debugging