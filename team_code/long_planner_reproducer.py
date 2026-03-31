#!/usr/bin/env python3
"""
Offline reproducer for the longitudinal planner (LongPlanner) pipeline.

Loads a ScenarioLogger v2 record (.json.gz), filters to frames where
plan_with_reasoning == True, reconstructs the planner inputs from the
logged primitives, and re-runs LongPlanner.run_step() for each selected frame.

Usage:
    python long_planner_reproducer.py records.json.gz [--frame N] [--plot]

    --frame N   : replay only the N-th matching frame (0-indexed); default: all
    --plot      : show the ST costmap + path plot for each replayed frame
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
from local_planner.longitudinal.long_planner import LongPlanner

# ---------------------------------------------------------------------------
# Proxy objects (duck-typed stand-ins for CARLA and internal dataclasses)
# ---------------------------------------------------------------------------

class _Location:
    def __init__(self, x, y, z):
        self.x = x; self.y = y; self.z = z

class _Extent:
    def __init__(self, x, y, z=0.0):
        self.x = x; self.y = y; self.z = z

class _Rotation:
    def __init__(self, yaw):
        self.yaw = yaw

class _BBoxProxy:
    """
    Duck-typed stand-in for carla.BoundingBox.

    Serialized format: [x, y, z, extent_x, extent_y, yaw_deg]
    bbox_to_vec7() inside LongPlanner.run_step() also reads .extent.z, so we
    default that to 0 (it is never used in the actual grid computations).
    """
    def __init__(self, vec6: list):
        self.location = _Location(vec6[0], vec6[1], vec6[2])
        self.extent   = _Extent(vec6[3], vec6[4], 0.0)
        self.rotation = _Rotation(vec6[5])


class _RouteSubsetBBoxProxy:
    """
    Minimal stand-in for the route-subset bounding boxes stored in
    LaneOverlapInterval.route_subset_bboxes.  Only .extent.x is read by
    STOccupancyGrid._build_s_bands_from_occupancies().
    """
    def __init__(self, extent_x: float):
        self.extent = _Extent(extent_x, 0.0)


class _LaneOverlapProxy:
    """
    Duck-typed stand-in for LaneOverlapInterval.

    Reconstructed from the serialized dict produced by ScenarioLogger
    (which now includes route_subset_extent_xs, tbb_frame_occupancies,
    ebb_frame_occupancies).
    """
    def __init__(self, log_dict: dict):
        self.is_valid        = log_dict.get("is_valid", False)
        self.time_start_idx  = log_dict.get("time_start_idx", -1)
        self.time_end_idx    = log_dict.get("time_end_idx",   -1)
        self.space_start_idx = log_dict.get("space_start_idx", -1)
        self.space_end_idx   = log_dict.get("space_end_idx",   -1)

        self.route_subset_bboxes = [
            _RouteSubsetBBoxProxy(ex)
            for ex in log_dict.get("route_subset_extent_xs", [])
        ]

        # tbb/ebb_frame_occupancies: Dict[int, Tuple[int, int]]
        # Logged as list of [frame_idx, bb_start, bb_end] triples.
        self.tbb_frame_occupancies = {
            int(entry[0]): (int(entry[1]), int(entry[2]))
            for entry in log_dict.get("tbb_frame_occupancies", [])
        }
        self.ebb_frame_occupancies = {
            int(entry[0]): (int(entry[1]), int(entry[2]))
            for entry in log_dict.get("ebb_frame_occupancies", [])
        }


class _CollisionIntervalProxy:
    """
    Duck-typed stand-in for CollisionInterval.

    Only collision_bboxes_b[start_idx] is accessed by LongPlanner.run_step()
    during the ego-rollout safety check.
    """
    def __init__(self, log_dict: dict):
        self.start_idx = log_dict["start_idx"]
        self.end_idx   = log_dict["end_idx"]
        self.collision_bboxes_b = [
            _BBoxProxy(v) for v in log_dict.get("collision_bboxes_b", [])
        ]


class _PredictionDataProxy:
    """Duck-typed stand-in for PredictionData."""
    def __init__(self, log_dict: dict):
        self.ego_forecasted_bbs = [
            _BBoxProxy(v) for v in log_dict.get("ego_forecasted_bbs", [])
        ]
        # veh_dilated_forecasted_bbs — not accessed by LongPlanner; set empty
        self.veh_dilated_forecasted_bbs: Dict = {}

        self.all_actor_overlaps: Dict[int, _LaneOverlapProxy] = {
            int(k): _LaneOverlapProxy(v)
            for k, v in log_dict.get("actor_overlaps", {}).items()
        }
        self.all_actor_collisions: Dict[int, List[_CollisionIntervalProxy]] = {
            int(k): [_CollisionIntervalProxy(ci) for ci in intervals]
            for k, intervals in log_dict.get("actor_collisions", {}).items()
        }


# ---------------------------------------------------------------------------
# Reconstruction helpers
# ---------------------------------------------------------------------------

def _build_planner_state(ps_log: dict) -> SimpleNamespace:
    """
    Reconstruct a PlannerState-like namespace from a logged planner_state dict.

    The log stores slices beginning at route_index, so we reset route_index to 0
    and use the sliced arrays as the full route.
    """
    route_pts = np.array(ps_log["route_points_ahead"], dtype=np.float32)
    s_route   = np.array(ps_log["s_route_ahead"],      dtype=np.float32)

    return SimpleNamespace(
        route_index=0,
        route_points=route_pts,
        s_route=s_route,
    )


def _build_scene_data(sc_log: dict) -> SimpleNamespace:
    """
    Reconstruct a SceneData-like namespace from the logged scene_context dict.
    LongPlanner accesses:
        scene_data.ego_data.speed
        scene_data.ego_data.accel
        scene_data.traffic_data.speed_limit
    """
    ego     = sc_log.get("ego",     {})
    traffic = sc_log.get("traffic", {})

    ego_data     = SimpleNamespace(
        speed=float(ego.get("speed", 0.0)),
        accel=float(ego.get("accel", 0.0)),
    )
    traffic_data = SimpleNamespace(
        speed_limit=float(traffic.get("speed_limit", 50.0 / 3.6)),
    )
    return SimpleNamespace(
        ego_data=ego_data,
        traffic_data=traffic_data,
    )


def _build_prediction_data(pred_log: dict) -> _PredictionDataProxy:
    return _PredictionDataProxy(pred_log)


def _build_all_conditions(ego_plan_log: dict) -> Dict[int, Tuple]:
    """
    Reconstruct the all_conditions dict that trajectory_planner passes to
    LongPlanner.  The 4-tuple is (condition_action, obj_type, traffic_type,
    priority).  STOP_FOR conditions targeting stop_sign / traffic_light /
    obstacle are excluded (they go into stop_conditions in trajectory_planner).
    """
    from scene_analyzer.parsers.ego_plan_pydantic_models import ConditionAction
    all_conditions: Dict[int, Tuple] = {}
    for cond in ego_plan_log.get("conditions", []):
        action = cond.get("condition_action", "")
        obj_type = cond.get("obj_type", "default")
        if action == ConditionAction.STOP_FOR or action == "stop_for":
            if obj_type in ("stop_sign", "traffic_light", "obstacle"):
                continue
        traffic_type = cond.get("traffic_type", "default")
        priority     = cond.get("priority",     "medium")
        all_conditions[cond["id"]] = (action, obj_type, traffic_type, priority)
    return all_conditions


def _build_ego_plan(ego_plan_log: dict):
    """Deserialize EgoPlan from logged dict (needed to print context only)."""
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
    long_planner: LongPlanner,
    plot: bool = False,
) -> None:
    """
    Replay one logged frame through LongPlanner.run_step().

    Args:
        records      : Full records dict from the log file.
        frame_idx    : Index into the records lists (not the raw simulation step).
        long_planner : A LongPlanner instance to run.
        plot         : Whether to display the ST costmap visualisation.
    """
    frame_number = records.get("frame_indices", [None])[frame_idx]
    print(f"\n{'='*60}")
    print(f"Replaying logged frame index {frame_idx}  (sim step {frame_number})")
    print(f"{'='*60}")

    ps_log   = records["planner_states"][frame_idx]
    sc_log   = records["scene_contexts"][frame_idx]
    pred_log = records.get("prediction_data", [{}])[frame_idx]
    ep_log   = records.get("ego_plans",        [{}])[frame_idx]

    planner_state   = _build_planner_state(ps_log)
    scene_data      = _build_scene_data(sc_log)
    prediction_data = _build_prediction_data(pred_log)
    all_conditions  = _build_all_conditions(ep_log)
    ego_plan        = _build_ego_plan(ep_log)

    n_overlaps   = len(prediction_data.all_actor_overlaps)
    n_collisions = sum(len(v) for v in prediction_data.all_actor_collisions.values())
    n_ego_bbs    = len(prediction_data.ego_forecasted_bbs)
    print(f"  Ego speed      : {scene_data.ego_data.speed:.2f} m/s")
    print(f"  Speed limit    : {scene_data.traffic_data.speed_limit:.2f} m/s")
    print(f"  Overlaps       : {n_overlaps} actors")
    print(f"  Collisions     : {n_collisions} intervals")
    print(f"  Ego forecast   : {n_ego_bbs} frames")
    if ego_plan is not None:
        print(f"  EgoPlan action : {ego_plan.action.value}")
        print(f"  Conditions     : {[(c.condition_action, c.id) for c in ego_plan.conditions]}")

    # Reset the planner so each frame is independent
    long_planner.reset_plan()

    t0 = time.perf_counter()
    result = long_planner.run_step(
        plan_tick_counter=1,          # force a planning tick (1 % any_freq == 1)
        planner_state=planner_state,
        scene_data=scene_data,
        prediction_data=prediction_data,
        all_conditions=all_conditions,
        plan_s_goal_m=40.0,           # use full ST grid horizon
        s_ego_m=0.0,
        ego_cruise_speed=None,        # defaults to speed_limit inside run_step
        ego_goal_speed=None,
        s_bounds=None,
        profile_time=True,
    )
    t1 = time.perf_counter()
    print(f"LongPlanner runtime: {(t1 - t0)*1000:.2f} ms")

    if result.is_empty_plan:
        print("  [Result] LongPlanner returned an empty plan.")
    else:
        print(f"  [Result] Plan: route_start={result.plan_route_start_idx}, "
              f"target_speed={result.target_speed:.2f} m/s, "
              f"cost={result.plan_cost:.2f}, "
              f"profile_len={len(result.vel_profile)}")

    if plot and result.st_maps is not None:
        long_planner.plot_st_map(
            long_planner.grid_mapper,
            result.st_maps.cost_map,
            scene_data.traffic_data.speed_limit,
        )
        import matplotlib.pyplot as plt
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="LongPlanner offline reproducer")
    parser.add_argument("records", help="Path to records.json.gz log file")
    parser.add_argument("--frame", type=int, default=None,
                        help="Replay only the N-th plan_with_reasoning frame (0-indexed)")
    parser.add_argument("--plot", action="store_true",
                        help="Show ST costmap plots")
    args = parser.parse_args()

    config      = GlobalConfig()
    long_planner = LongPlanner(config=config)

    records = load_records(args.records)

    reasoning_frame_indices = get_reasoning_frames(records)
    print(f"Found {len(reasoning_frame_indices)} plan_with_reasoning frames "
          f"out of {len(records.get('plan_with_reasoning', []))} total logged frames.")

    if args.frame is not None:
        if args.frame >= len(reasoning_frame_indices):
            print(f"[Error] --frame {args.frame} out of range "
                  f"(only {len(reasoning_frame_indices)} matching frames).")
            sys.exit(1)
        target_indices = [reasoning_frame_indices[args.frame]]
    else:
        # target_indices = reasoning_frame_indices
        target_indices = range(0, len(records.get('plan_with_reasoning', [])))

    for idx in target_indices:
        replay_frame(records, idx, long_planner, plot=args.plot)


if __name__ == "__main__":
    main()
