#!/usr/bin/env python3
"""
Offline reproducer for the lateral planner (LatPlanner) pipeline.

Loads a ScenarioLogger v2 record (.json.gz), filters to frames where
plan_with_reasoning == True, reconstructs the planner inputs from the
logged primitives, and re-runs LatPlanner.run_step() for each selected frame.

Usage:
    python lat_planner_reproducer.py records.json.gz [--frame N] [--plot]

    --frame N   : replay only the N-th matching frame (0-indexed); default: all
    --plot      : show the SL costmap + path plot for each replayed frame
"""

import argparse
import gzip
import json
import sys
import numpy as np
import time

from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

from config import GlobalConfig
from local_planner.lateral.lat_planner import LatPlanner

# ---------------------------------------------------------------------------
# Proxy objects (duck-typed stand-ins for CARLA and internal dataclasses)
# ---------------------------------------------------------------------------

class _WaypointProxy:
    """Minimal stand-in for carla.Waypoint exposing only what LatPlanner needs."""
    def __init__(self, lane_width: float, left_lane_width: Optional[float] = None,
                 right_lane_width: Optional[float] = None):
        self.lane_width = lane_width
        self._left_lane_width = left_lane_width
        self._right_lane_width = right_lane_width

    def get_left_lane(self) -> Optional["_WaypointProxy"]:
        if self._left_lane_width is None:
            return None
        return _WaypointProxy(self._left_lane_width)

    def get_right_lane(self) -> Optional["_WaypointProxy"]:
        if self._right_lane_width is None:
            return None
        return _WaypointProxy(self._right_lane_width)


class _WaypointListProxy:
    """
    List-like proxy for PlannerState.route_waypoints.
    Only index[0] (the current position) is meaningful; all other indices
    return a default waypoint with the same lane width.
    """
    def __init__(self, lane_width: float, left_lane_width: Optional[float],
                 right_lane_width: Optional[float]):
        self._wp = _WaypointProxy(lane_width, left_lane_width, right_lane_width)

    def __getitem__(self, idx):
        return self._wp


class _ObstacleProxy:
    """Duck-typed stand-in for ObstacleDataEntry exposing .id and .bbox_corners_xy."""
    def __init__(self, obs_id: int, relative_distance : float, bbox_corners_xy: np.ndarray):
        self.id = obs_id
        self.relative_distance = relative_distance
        self.bbox_corners_xy = bbox_corners_xy


class _ObstacleDataProxy:
    """Duck-typed stand-in for ObstacleData."""
    def __init__(self, all_obstacles: List[_ObstacleProxy]):
        self.all_obstacles = all_obstacles


# ---------------------------------------------------------------------------
# Reconstruction helpers
# ---------------------------------------------------------------------------

def _build_planner_state(ps_log: dict) -> SimpleNamespace:
    """
    Reconstruct a PlannerState-like namespace from a logged planner_state dict.

    The log stores slices beginning at route_index, so we reset route_index to 0
    and route_len to the slice length.
    """
    route_pts = np.array(ps_log["route_points_ahead"], dtype=np.float32)
    rot_angles = np.array(ps_log["rotation_angles_ahead"], dtype=np.float32)
    route_cmds = np.array(ps_log["route_commands_ahead"], dtype=np.float32)
    s_route = np.array(ps_log["s_route_ahead"], dtype=np.float32)

    n = len(route_pts)

    return SimpleNamespace(
        route_index=0,
        route_len=n,
        original_route_points=route_pts,
        original_rotation_angles=rot_angles,
        route_commands=route_cmds,
        original_route_s=s_route,
        route_waypoints=_WaypointListProxy(
            lane_width=ps_log.get("lane_width_current") or 3.5,
            left_lane_width=ps_log.get("lane_width_left"),
            right_lane_width=ps_log.get("lane_width_right"),
        ),
        original_route_waypoints=_WaypointListProxy(
            lane_width=ps_log.get("lane_width_current") or 3.5,
            left_lane_width=ps_log.get("lane_width_left"),
            right_lane_width=ps_log.get("lane_width_right"),
        ),
    )


def _build_lane_info(sc_log: dict) -> SimpleNamespace:
    """
    Reconstruct a LaneInfo-like namespace from the logged scene_context dict.
    """
    rd = sc_log.get("route_data", {})
    li = rd.get("lane_info", {})

    lw_left = li.get("lane_width_left")
    lw_right = li.get("lane_width_right")

    left_wp = _WaypointProxy(lw_left) if lw_left is not None else None
    right_wp = _WaypointProxy(lw_right) if lw_right is not None else None

    ns = SimpleNamespace(
        has_left_lane=li.get("has_left_lane", False),
        has_right_lane=li.get("has_right_lane", False),
        left_oncoming=li.get("left_oncoming", False),
        right_oncoming=li.get("right_oncoming", False),
        left_same_dir=li.get("left_same_dir", False),
        right_same_dir=li.get("right_same_dir", False),
        left_wp=left_wp,
        right_wp=right_wp,
    )
    # Mirror the @property from LaneInfo
    ns.same_direction_lane_change_available = ns.left_same_dir or ns.right_same_dir
    return ns


def _build_scene_data(sc_log: dict) -> SimpleNamespace:
    """
    Reconstruct a SceneData-like namespace from the logged scene_context dict.
    LatPlanner only accesses scene_data.route_data.lane_info and
    scene_data.obstacle_data.all_obstacles.
    """
    lane_info = _build_lane_info(sc_log)
    route_data = SimpleNamespace(lane_info=lane_info)

    obs_list = []
    for obs in sc_log.get("obstacles", []):
        raw_corners = obs.get("bbox_corners_xy")
        if raw_corners is None:
            continue
        corners_arr = np.array(raw_corners, dtype=np.float32)
        obs_list.append(_ObstacleProxy(obs_id=obs["id"], relative_distance=obs["relative_distance"], bbox_corners_xy=corners_arr))

    obstacle_data = _ObstacleDataProxy(all_obstacles=obs_list)

    return SimpleNamespace(
        route_data=route_data,
        obstacle_data=obstacle_data,
    )


def _build_all_conditions(ego_plan_log: dict) -> Dict[int, Tuple]:
    """
    Reconstruct the all_conditions dict that trajectory_planner passes to LatPlanner.
    Only non-stop_for conditions are included (mirrors trajectory_planner logic).
    """
    all_conditions = {}
    for cond in ego_plan_log.get("conditions", []):
        if cond.get("condition_action") == "stop_for":
            continue
        actor_id = cond["id"]
        action_type = cond["condition_action"]   # e.g. "yield_for"
        actor_type = cond["obj_type"]            # e.g. "vehicle"
        importance = cond.get("importance", 1.0)
        all_conditions[actor_id] = (action_type, actor_type, float(importance))
    return all_conditions


def _build_ego_plan(ego_plan_log: dict):
    """Deserialize EgoPlan from logged dict."""
    if not ego_plan_log:
        return None
    try:
        from scene_analyzer.parsers.ego_plan_pydantic_models import EgoPlan
        return EgoPlan.model_validate(ego_plan_log)
    except Exception as exc:
        print(f"[Warn] Could not deserialize EgoPlan: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main reproducer logic
# ---------------------------------------------------------------------------

def load_records(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def get_reasoning_frames(records: dict) -> List[int]:
    """Return indices (into the records lists) where plan_with_reasoning is True."""
    flags = records.get("plan_with_reasoning", [])
    return [i for i, flag in enumerate(flags) if flag]


def replay_frame(
    records: dict,
    frame_idx: int,
    lat_planner : LatPlanner,
    plot: bool = False,
) -> None:
    """
    Replay one logged frame through LatPlanner.run_step().

    Args:
        records   : Full records dict from the log file.
        frame_idx : Index into the records lists (not the raw simulation step).
        lat_planner : A LatPlanner instance to run.
        plot      : Whether to display the SL costmap visualisation.
    """
    frame_number = records.get("frame_indices", [None])[frame_idx]
    print(f"\n{'='*60}")
    print(f"Replaying logged frame index {frame_idx}  (sim step {frame_number})")
    print(f"{'='*60}")

    ps_log  = records["planner_states"][frame_idx]
    sc_log  = records["scene_contexts"][frame_idx]
    ep_log  = records.get("ego_plans", [{}])[frame_idx]

    planner_state  = _build_planner_state(ps_log)
    scene_data     = _build_scene_data(sc_log)
    ego_plan       = _build_ego_plan(ep_log)
    all_conditions = _build_all_conditions(ep_log)

    if ego_plan is None:
        print("[Skip] No valid EgoPlan for this frame — skipping.")
        return

    print(f"  EgoPlan action : {ego_plan.action.value}")
    print(f"  Conditions     : {[(c.condition_action.value, c.id) for c in ego_plan.conditions]}")
    print(f"  Obstacles      : {len(scene_data.obstacle_data.all_obstacles)}")

    # Reset the planner so each frame is independent
    lat_planner.reset_plan()

    t0 = time.perf_counter()
    result = lat_planner.run_step(
        plan_tick_counter=1,          # force a planning tick (1 % any_freq == 1)
        planner_state=planner_state,
        lidar_data={},
        scene_data=scene_data,
        prediction_data=None,
        ego_plan=ego_plan,
        all_conditions=all_conditions,
        profile_time=True,
    )
    t1 = time.perf_counter()
    print(f"LatPlanner runtime: {(t1 - t0)*1000:.2f} ms")

    if result.is_empty_plan:
        print("  [Result] LatPlanner returned an empty plan.")
    else:
        print(f"  [Result] Plan: start={result.start_idx}, goal={result.goal_idx}, "
              f"s_distance={result.s_distance:.1f} m, "
              f"route_pts={len(result.route_points)}")

    if plot and result.sl_maps is not None:
        lat_planner.plot_path_and_profiles(
            sl_maps=result.sl_maps,
            dp_path=result.dp_path if result.dp_path is not None else np.zeros((0, 2)),
            corridor=result.corridor if result.corridor is not None else np.zeros((0, 2)),
            qp_path=result.qp_path if result.qp_path is not None else np.zeros((0, 2)),
        )


def main():
    parser = argparse.ArgumentParser(description="LatPlanner offline reproducer")
    parser.add_argument("records", help="Path to records.json.gz log file")
    parser.add_argument("--frame", type=int, default=None,
                        help="Replay only the N-th plan_with_reasoning frame (0-indexed)")
    parser.add_argument("--plot", action="store_true",
                        help="Show SL costmap + path plots")
    args = parser.parse_args()

    config = GlobalConfig()
    lat_planner = LatPlanner(config=config)

    records = load_records(args.records)

    reasoning_frame_indices = get_reasoning_frames(records)
    print(f"Found {len(reasoning_frame_indices)} plan_with_reasoning frames "
          f"out of {len(records.get('plan_with_reasoning', []))} total logged frames.")

    # if not reasoning_frame_indices:
    #     print("No plan_with_reasoning frames found — nothing to reproduce.")
    #     return

    if args.frame is not None:
        if args.frame >= len(reasoning_frame_indices):
            print(f"[Error] --frame {args.frame} out of range "
                  f"(only {len(reasoning_frame_indices)} matching frames).")
            sys.exit(1)
        target_indices = [reasoning_frame_indices[args.frame]]
    else:
        # target_indices = reasoning_frame_indices
        target_indices = [17]

    for idx in target_indices:
        replay_frame(records, idx, lat_planner, plot=args.plot)


if __name__ == "__main__":
    main()
