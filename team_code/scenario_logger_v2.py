"""
Creates log files during evaluation with which we can visualize failures.
Logs simulation state, PlannerState, SceneContext, PredictionData, and
planner results per step.
"""

import os
import json
import carla
import gzip
import numpy as np
from rdp import rdp

from typing import Optional, TYPE_CHECKING

# TODO: REMOVE DUPLICATED WORK IN LOGGER

class _NumpyEncoder(json.JSONEncoder):
    """Converts numpy scalars and arrays to native Python types for json.dump."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

if TYPE_CHECKING:
    from privileged_route_planner import PlannerState
    from scene_descriptor.scene_descriptor import SceneContext
    from actor_prediction.motion_prediction import PredictionData


class ScenarioLogger:
    """
    Creates log files during evaluation with which we can visualize failures.
    Logs vehicle/traffic/route/control state alongside structured planner data.
    """

    def __init__(self, save_path, route_index, logging_freq, log_only, route_only, roi=30, rdp_epsilon=0.5) -> None:
        # logger settings
        self.logging_freq = logging_freq
        self.log_only = log_only
        self.route_only = route_only

        self.roi = roi              # radius around ego agent in meters
        self.rdp_epsilon = rdp_epsilon

        # meta data
        self.save_path = save_path
        self.route_index = route_index

        # simulation objects (set externally before first log_step call)
        self.world = None
        self.ego_vehicle = None
        self.step = 0

        # Per-frame output directory (created on first write)
        self.frames_dir = os.path.join(str(save_path), "lon_log_frames")
        self.frame_count = 0

        # ------------------------------------------------------------------ #
        # Per-step working buffers (reset after each logged step)
        # ------------------------------------------------------------------ #

        self.ego_location = None

        self.ego_pos    = None
        self.ego_yaw    = None
        self.ego_vel    = None
        self.ego_extent = None
        self.ego_id     = None
        self.ego_type   = None
        self.ego_color  = None

        self.bg_vehicles = []
        self.bg_pos    = None
        self.bg_yaw    = None
        self.bg_vel    = None
        self.bg_extent = None
        self.bg_id     = None
        self.bg_type   = None
        self.bg_color  = None

        self.tlights   = []
        self.tl_pos    = None
        self.tl_yaw    = None
        self.tl_state  = None
        self.tl_extent = None

        # Route boxes are cached between steps and only recomputed when the
        # route index advances by more than _ROUTE_CACHE_STRIDE points.
        self._cached_route_ri = -9999
        self.route_pos    = None
        self.route_yaw    = None
        self.route_id     = None
        self.route_extent = None

    # ---------------------------------------------------------------------- #
    # Simulation state helpers (unchanged from original ScenarioLogger)
    # ---------------------------------------------------------------------- #

    def _initialize_bg_agents(self):
        actors = self.world.get_actors()

        vehicles = actors.filter("*vehicle*")
        self.bg_vehicles = []
        for vehicle in vehicles:
            if vehicle.id != self.ego_vehicle.id:
                vehicle_location = vehicle.get_transform().location
                if vehicle_location.distance(self.ego_location) < self.roi:
                    self.bg_vehicles.append(vehicle)

        tlights = actors.filter("*traffic_light*")
        self.tlights = []
        for tlight in tlights:
            if tlight.state != carla.libcarla.TrafficLightState.Green:
                trigger_box_global_pos = tlight.get_transform().transform(tlight.trigger_volume.location)
                trigger_box_global_pos = carla.Location(
                    x=trigger_box_global_pos.x,
                    y=trigger_box_global_pos.y,
                    z=trigger_box_global_pos.z,
                )
                if trigger_box_global_pos.distance(self.ego_location) < self.roi:
                    self.tlights.append(tlight)

    def fetch_bg_state(self):
        positions, yaws, velocities, extents, ids, types, colors = [], [], [], [], [], [], []

        for vehicle in self.bg_vehicles:
            t  = vehicle.get_transform()
            v  = vehicle.get_velocity()
            bb = vehicle.bounding_box.extent
            ey, ex = bb.y, bb.x

            positions.append([t.location.x, t.location.y])
            yaws.append(np.radians(t.rotation.yaw))
            velocities.append([v.x, v.y])
            extents.append([[ey, ex], [ey, -ex], [-ey, -ex], [-ey, ex]])
            ids.append(vehicle.id)
            types.append(vehicle.type_id)
            try:
                colors.append(vehicle.attributes["color"])
            except KeyError:
                colors.append("0,0,0")

        n = len(self.bg_vehicles)
        if n > 0:
            self.bg_pos    = np.array(positions,  dtype=float).reshape(1, n, 2)
            self.bg_yaw    = np.array(yaws,        dtype=float).reshape(1, n, 1)
            self.bg_vel    = np.array(velocities,  dtype=float).reshape(1, n, 2)
            self.bg_extent = np.array(extents,     dtype=float).reshape(1, n, 4, 2)
            self.bg_id     = np.array(ids).reshape(1, n, 1)
            self.bg_type   = np.array(types).reshape(1, n, 1)
            self.bg_color  = np.array(colors).reshape(1, n, 1)

        # traffic lights
        tl_positions, tl_yaws, tl_states, tl_extents = [], [], [], []
        for tlight in self.tlights:
            if tlight.state == carla.libcarla.TrafficLightState.Red:
                state = 0
            elif tlight.state == carla.libcarla.TrafficLightState.Yellow:
                state = 1
            else:
                continue

            center = tlight.get_transform().transform(tlight.trigger_volume.location)
            center = carla.Location(center.x, center.y, center.z)
            extent_vec = carla.Vector3D(
                tlight.trigger_volume.extent.x,
                tlight.trigger_volume.extent.y,
                tlight.trigger_volume.extent.z,
            )
            transform = carla.Transform(center)
            bounding_box = carla.BoundingBox(transform.location, extent_vec)
            global_rot = tlight.get_transform().rotation
            bounding_box.rotation = carla.Rotation(
                pitch=tlight.trigger_volume.rotation.pitch + global_rot.pitch,
                yaw=tlight.trigger_volume.rotation.yaw + global_rot.yaw,
                roll=tlight.trigger_volume.rotation.roll + global_rot.roll,
            )

            tl_positions.append(np.array([[[center.x, center.y]]]))
            tl_yaws.append(np.array([[[bounding_box.rotation.yaw]]]))
            tl_states.append(np.array([[[state]]]))
            tl_extents.append(
                np.array([[[
                    [bounding_box.extent.y, bounding_box.extent.x],
                    [bounding_box.extent.y, -bounding_box.extent.x],
                    [-bounding_box.extent.y, -bounding_box.extent.x],
                    [-bounding_box.extent.y, bounding_box.extent.x],
                ]]]))

        if self.tl_pos is None and tl_positions:
            self.tl_pos = np.concatenate(tl_positions, axis=1)
        if self.tl_yaw is None and tl_yaws:
            self.tl_yaw = np.concatenate(np.radians(tl_yaws), axis=1)
        if self.tl_state is None and tl_states:
            self.tl_state = np.concatenate(tl_states, axis=1)
        if self.tl_extent is None and tl_extents:
            self.tl_extent = np.concatenate(tl_extents, axis=1)

    # How many route-index steps must pass before the cached route boxes are rebuilt.
    _ROUTE_CACHE_STRIDE = 20

    def route_as_boxes(self, route_xy, current_ri: int):
        """
        Convert a 2-D route array (N x 2) to route bounding boxes using RDP
        simplification.  Results are cached; recomputation only happens when
        the route index has advanced by at least _ROUTE_CACHE_STRIDE points.
        """
        if abs(current_ri - self._cached_route_ri) < self._ROUTE_CACHE_STRIDE \
                and self.route_pos is not None:
            return  # use cached result

        shortened_route = rdp(route_xy, epsilon=self.rdp_epsilon)
        if len(shortened_route) < 2:
            return

        vectors   = shortened_route[1:] - shortened_route[:-1]
        midpoints = shortened_route[:-1] + vectors / 2.0
        norms     = np.linalg.norm(vectors, axis=1)
        angles    = np.arctan2(vectors[:, 1], vectors[:, 0])   # radians

        # Ego half-width used as the lateral extent of each route box
        ey = float(self.ego_vehicle.bounding_box.extent.y)
        ego_xy = np.array([self.ego_location.x, self.ego_location.y])

        positions, yaws, ids, extents = [], [], [], []
        for i, midpoint in enumerate(midpoints):
            start = shortened_route[i]
            if 0 < i < 10 and np.linalg.norm(start - ego_xy) > self.roi:
                continue

            ex = norms[i] / 2.0   # half-length
            positions.append(midpoint)
            yaws.append(angles[i])
            ids.append(i)
            extents.append([[ey, ex], [ey, -ex], [-ey, -ex], [-ey, ex]])

        if positions:
            n = len(positions)
            self.route_pos    = np.array(positions, dtype=float).reshape(1, n, 2)
            self.route_yaw    = np.array(yaws,      dtype=float).reshape(1, n, 1)
            self.route_id     = np.array(ids).reshape(1, n, 1)
            self.route_extent = np.array(extents,   dtype=float).reshape(1, n, 4, 2)
            self._cached_route_ri = current_ri

    # ---------------------------------------------------------------------- #
    # Structured data serializers
    # ---------------------------------------------------------------------- #

    def _serialize_planner_state(self, planner_state) -> dict:
        """
        Serialize a PlannerState to a JSON-compatible dict.
        Carla objects (Waypoint, TrafficLight, BoundingBox) are skipped or
        reduced to primitive fields.
        """
        if planner_state is None:
            return {}

        # Route commands: RoadOption enum → int value
        route_commands = (
            [int(c.value) if hasattr(c, 'value') else int(c)
             for c in planner_state.route_commands]
            if planner_state.route_commands is not None else []
        )

        # Nearby route points only (limit to roi around current index)
        route_pts = planner_state.route_points
        ri = planner_state.route_index
        look = min(len(route_pts) - ri, 501)   # up to 501 (~50m) points ahead
        route_pts_slice = route_pts[ri:ri + look].tolist() if route_pts is not None else []

        s_route = planner_state.s_route
        s_route_slice = s_route[ri:ri + look].tolist() if s_route is not None else []

        rotation_angles = planner_state.rotation_angles
        rotation_angles_slice = rotation_angles[ri:ri + look].tolist() if rotation_angles is not None else []

        return {
            "route_index":             planner_state.route_index,
            "route_len":               planner_state.route_len,
            "route_points_ahead":      route_pts_slice,
            "rotation_angles_ahead":   rotation_angles_slice,
            "route_commands_ahead":    route_commands[ri:ri + look],
            "s_route_ahead":           s_route_slice,
            "speed_limits": (
                planner_state.speed_limits[:look].tolist()
                if planner_state.speed_limits is not None else []
            ),
            "dist_to_next_traffic_lights": (
                planner_state.dist_to_next_traffic_lights.tolist()
                if planner_state.dist_to_next_traffic_lights is not None else []
            ),
            "dist_to_next_stop_signs": (
                planner_state.dist_to_next_stop_signs.tolist()
                if planner_state.dist_to_next_stop_signs is not None else []
            ),
            "cleared_stop_sign_ids": list(planner_state.cleared_stop_sign_ids),
        }

    def _serialize_scene_context(self, scene_context) -> dict:
        """
        Serialize a SceneContext to a JSON-compatible dict.
        Carla actor objects are reduced to primitive fields.
        """
        if scene_context is None:
            return {}

        result = {
            "formatted_text": scene_context.formatted_text,
            "ego": {},
            "traffic": {},
            "vehicles": [],
            "pedestrians": [],
            "obstacles": [],
            "collisions": [],
        }

        scene_data = scene_context.scene_data

        # Ego data
        if scene_data.ego_data is not None:
            ed = scene_data.ego_data
            result["ego"] = {
                "speed": ed.speed,
                "orientation": ed.orientation,
                "position": ed.position,
                "accel": ed.accel,
            }

        # Traffic data
        if scene_data.traffic_data is not None:
            td = scene_data.traffic_data
            tl = td.next_traffic_light
            ss = td.next_stop_sign
            result["traffic"] = {
                "speed_limit": td.speed_limit,
                "next_traffic_light": (
                    {
                        "id": tl.id,
                        "distance": tl.distance_to_light,
                        "state": tl.state,
                    }
                    if tl is not None else None
                ),
                "next_stop_sign": (
                    {
                        "id": ss.id,
                        "distance": ss.distance_to_stop_sign,
                        "cleared": ss.cleared,
                    }
                    if ss is not None else None
                ),
            }

        # Vehicle data
        if scene_data.vehicle_data is not None:
            for v in scene_data.vehicle_data.all_vehicles_flat:
                result["vehicles"].append({
                    "id": v.id,
                    "vehicle_type": v.vehicle_type,
                    "traffic_type": v.traffic_type,
                    "lane_name": v.lane_name,
                    "lanelet_id": v.lanelet_id,
                    "lanelet_route_idx": v.lanelet_route_idx,
                    "x": v.x,
                    "y": v.y,
                    "z": v.z,
                    "heading": v.heading,
                    "speed": v.speed,
                    "relative_orientation": v.relative_orientation,
                    "relative_position": list(v.relative_position),
                    "relative_distance": v.relative_distance,
                    "intrudes_ego": v.intrudes_ego,
                    "intrusion_idx": v.intrusion_idx,
                    "emergency_sirens_active": v.emergency_sirens_active,
                    "throttle": v.throttle,
                    "steer": v.steer,
                    "brake": v.brake,
                })

        # Pedestrian data
        if scene_data.ped_data is not None:
            for p in scene_data.ped_data:
                result["pedestrians"].append({
                    "id": p.id,
                    "speed": p.speed,
                    "relative_orientation": p.relative_orientation,
                    "relative_position": list(p.relative_position),
                    "relative_distance": p.relative_distance,
                    "is_on_road": p.is_on_road,
                })

        # Obstacle data
        if scene_data.obstacle_data is not None:
            for obs in scene_data.obstacle_data.all_obstacles:
                obs_entry = {
                    "id": obs.id,
                    "relative_position": list(obs.relative_position),
                    "relative_distance": obs.relative_distance,
                    "obstructs_ego": obs.obstructs_ego,
                    "intrusion_idx": obs.intrusion_idx,
                    "is_near_junction": obs.is_near_junction,
                }
                if obs.bbox_corners_xy is not None:
                    obs_entry["bbox_corners_xy"] = obs.bbox_corners_xy.tolist()
                result["obstacles"].append(obs_entry)

        # Route data (lane info for lateral planner)
        if scene_data.route_data is not None:
            li = scene_data.route_data.lane_info
            result["route_data"] = {
                "lane_info": {
                    "has_left_lane": li.has_left_lane,
                    "has_right_lane": li.has_right_lane,
                    "left_oncoming": li.left_oncoming,
                    "right_oncoming": li.right_oncoming,
                    "left_same_dir": li.left_same_dir,
                    "right_same_dir": li.right_same_dir,
                    "lane_width_left": float(li.left_wp.lane_width) if li.left_wp is not None else None,
                    "lane_width_right": float(li.right_wp.lane_width) if li.right_wp is not None else None,
                }
            }

        # Collision data
        if scene_data.collision_data:
            for c in scene_data.collision_data:
                try:
                    actor_id = c.actor.id
                except Exception:
                    actor_id = None
                result["collisions"].append({
                    "actor_id": actor_id,
                    "timestamp": c.timestamp,
                    "intensity": c.intensity,
                })

        return result

    @staticmethod
    def _serialize_bbox(bb) -> list:
        """
        Serialize a carla.BoundingBox to a compact list.

        Format: [x, y, z, extent_x, extent_y, yaw_deg]
          x, y, z      – world-frame centre (location)
          extent_x     – half-length along the actor's forward axis
          extent_y     – half-width  along the actor's lateral axis
          yaw_deg      – heading in degrees (CARLA convention)

        The visualizer reconstructs corners as:
          local = [[ ey,  ex], [ ey, -ex], [-ey, -ex], [-ey,  ex]]
          then rotated by yaw_deg around the centre.
        """
        return [
            bb.location.x, bb.location.y, bb.location.z,
            bb.extent.x,   bb.extent.y,
            bb.rotation.yaw,
        ]

    def _serialize_prediction_data(self, prediction_data) -> dict:
        """
        Serialize a PredictionData to a JSON-compatible dict.

        Bounding boxes for every predicted frame are stored so the visualizer
        can draw them directly and use the collision/overlap interval indices
        to identify which frames to highlight without any post-hoc projection.
        """
        if prediction_data is None:
            return {}

        # ── predicted bounding boxes ────────────────────────────────────────
        ego_bbs = [
            self._serialize_bbox(bb)
            for bb in (prediction_data.ego_forecasted_bbs or [])
        ]

        veh_bbs = {
            str(actor_id): [self._serialize_bbox(bb) for bb in bbs]
            for actor_id, bbs in (prediction_data.veh_forecasted_bbs or {}).items()
        }

        ped_bbs = {
            str(actor_id): [self._serialize_bbox(bb) for bb in bbs]
            for actor_id, bbs in (prediction_data.ped_forecasted_bbs or {}).items()
        }

        # ── collision intervals: {actor_id: [{start_idx, end_idx, bboxes_b}, ...]} ─
        collisions = {}
        if prediction_data.all_actor_collisions:
            for actor_id, intervals in prediction_data.all_actor_collisions.items():
                collisions[str(actor_id)] = [
                    {
                        "start_idx": ci.start_idx,
                        "end_idx":   ci.end_idx,
                        # Serialize collision_bboxes_b so the LongPlanner reproducer
                        # can reconstruct ego-rollout collision checks.
                        "collision_bboxes_b": [
                            self._serialize_bbox(bb)
                            for bb in (ci.collision_bboxes_b or [])
                        ],
                    }
                    for ci in intervals
                ]

        # ── lane overlap intervals: {actor_id: {is_valid, time/space idx, occ}} ─
        overlaps = {}
        if prediction_data.all_actor_overlaps:
            for actor_id, ov in prediction_data.all_actor_overlaps.items():
                # route_subset_bboxes: only extent.x is needed by STOccupancyGrid
                route_extent_xs = []
                try:
                    route_extent_xs = [
                        float(bb.extent.x) for bb in (ov.route_subset_bboxes or [])
                    ]
                except Exception:
                    pass

                # tbb/ebb_frame_occupancies: Dict[int, Tuple[int, int]]
                # Serialized as list of [frame_idx, bb_start, bb_end] triples.
                occ_to_list = lambda occ_dict: [
                    [int(k), int(v[0]), int(v[1])]
                    for k, v in (occ_dict or {}).items()
                ]

                overlaps[str(actor_id)] = {
                    "is_valid":             ov.is_valid,
                    "time_start_idx":       ov.time_start_idx,
                    "time_end_idx":         ov.time_end_idx,
                    "space_start_idx":      ov.space_start_idx,
                    "space_end_idx":        ov.space_end_idx,
                    "route_subset_extent_xs":  route_extent_xs,
                    "tbb_frame_occupancies":   occ_to_list(ov.tbb_frame_occupancies),
                    "ebb_frame_occupancies":   occ_to_list(ov.ebb_frame_occupancies),
                }

        return {
            "ego_forecasted_bbs": ego_bbs,
            "veh_forecasted_bbs": veh_bbs,
            "ped_forecasted_bbs": ped_bbs,
            "actor_collisions":   collisions,
            "actor_overlaps":     overlaps,
        }

    def _serialize_ego_plan(self, ego_plan) -> dict:
        """Serialize an EgoPlan pydantic model to a JSON-compatible dict."""
        if ego_plan is None:
            return {}
        try:
            d = ego_plan.model_dump()
            d["_text"] = ego_plan.to_string()
            return d
        except Exception:
            return {}

    def _serialize_high_level_beh(self, hl_beh) -> dict:
        """Serialize a HighLevelBehaviour pydantic model to a JSON-compatible dict."""
        if hl_beh is None:
            return {}
        try:
            d = hl_beh.model_dump()
            d["_text"] = hl_beh.to_string()
            return d
        except Exception:
            return {}

    def _serialize_long_planner_result(self, result) -> dict:
        """
        Serialize a LongPlannerResult to a JSON-compatible dict.

        Fields serialized:
            target_speed / plan_cost / plan_route_start_idx / has_active_plan – scalars
            vel_profile / time_profile / s_profile – planned kinematic profile
            plan_indices – (N, 2) int array of [t_idx, s_idx] grid coordinates
            st_maps – S_arr, T_arr, occupancy_map (T×S), cost_map (T×S)
        """
        if result is None:
            return {}

        vel     = getattr(result, "vel_profile",  np.array([]))
        time    = getattr(result, "time_profile", np.array([]))
        s       = getattr(result, "s_profile",    np.array([]))
        indices = getattr(result, "plan_indices", np.empty((0, 2), dtype=int))

        out = {
            "target_speed":         float(getattr(result, "target_speed", 0.0)),
            "plan_cost":            float(getattr(result, "plan_cost", float("inf"))),
            "plan_route_start_idx": int(getattr(result, "plan_route_start_idx", -1)),
            "has_active_plan":      bool(getattr(result, "has_active_plan", False)),
            "profile_step_idx":     int(getattr(result, "profile_step_idx", 0)),
            "vel_profile":          vel.tolist()     if isinstance(vel,     np.ndarray) else [],
            "time_profile":         time.tolist()    if isinstance(time,    np.ndarray) else [],
            "s_profile":            s.tolist()       if isinstance(s,       np.ndarray) else [],
            "plan_indices":         indices.tolist() if isinstance(indices, np.ndarray) else [],
        }

        # ── ST grid maps ─────────────────────────────────────────────────────
        st = getattr(result, "st_maps", None)
        if st is not None:
            out["st_maps"] = {
                "S_arr":          st.S_arr.tolist(),
                "T_arr":          st.T_arr.tolist(),
                "occupancy_map":  st.occupancy_map.tolist(),   # (T, S)
                "cost_map":       st.cost_map.tolist(),        # (T, S)
            }

        return out

    def _serialize_lat_planner_result(self, result) -> dict:
        """
        Serialize a LatPlannerResult to a JSON-compatible dict.

        Fields serialized:
            start_idx / goal_idx / s_distance / is_new_plan – scalar metadata
            route_points / route_yaws – planned path in world coordinates
            sl_maps  – S_arr, L_arr, L_ref, cost_map (2-D float array)
            dp_path  – (N, 2) Dijkstra seed [s, l]
            corridor – (N, 2) convex corridor [lb, ub]
            qp_path  – (M, 2) QP-optimised spline [s, l]
        """
        if result is None:
            return {}

        route_pts  = result.route_points  if isinstance(result.route_points,  np.ndarray) else np.array([])
        route_yaws = result.route_yaws    if isinstance(result.route_yaws,    np.ndarray) else np.array([])

        out = {
            "start_idx":   int(result.start_idx),
            "goal_idx":    int(result.goal_idx),
            "s_distance":  float(result.s_distance),
            "is_new_plan": bool(result.is_new_plan),
            "route_points": route_pts.tolist(),
            "route_yaws":   route_yaws.tolist(),
        }

        # ── SL grid maps ─────────────────────────────────────────────────────
        sl = result.sl_maps
        if sl is not None:
            out["sl_maps"] = {
                "S_arr":    sl.S_arr.tolist(),
                "L_arr":    sl.L_arr.tolist(),
                "L_ref":    float(sl.L_ref),
                "cost_map": sl.cost_map.tolist(),   # list[list[float]], shape (S, L)
            }

        # ── DP / corridor / QP paths ─────────────────────────────────────────
        if result.dp_path is not None and result.dp_path.size:
            out["dp_path"] = result.dp_path.tolist()
        if result.corridor is not None and result.corridor.size:
            out["corridor"] = result.corridor.tolist()
        if result.qp_path is not None and result.qp_path.size:
            out["qp_path"] = result.qp_path.tolist()

        return out

    # ---------------------------------------------------------------------- #
    # Main logging entry point
    # ---------------------------------------------------------------------- #

    def log_step(
        self,
        planner_state,
        scene_context,
        prediction_data=None,
        long_planner_result=None,
        lat_planner_result=None,
        ego_control=None,
        ego_plan=None,
        high_level_beh=None,
        plan_with_reasoning: bool = False,
        force: bool = False,
    ):
        """
        Record one simulation step.

        Args:
            planner_state        : PlannerState from PrivilegedRoutePlanner / TrajectoryPlanner
            scene_context        : SceneContext from SceneDescriptor
            prediction_data      : PredictionData from MotionPrediction (optional)
            long_planner_result  : LongPlanner result object (optional placeholder)
            lat_planner_result   : LatPlannerResult from LatPlanner (optional placeholder)
            ego_control          : carla.VehicleControl for the ego vehicle (optional)
            ego_plan             : EgoPlan from SceneAnalyzer / agent (optional)
            high_level_beh       : HighLevelBehaviour from SceneAnalyzer (optional)
            plan_with_reasoning  : bool returned by TrajectoryPlanner.plan_with_reasoning()
            force                : if True, bypass logging_freq and always record this step
                                   (use for collision events)
        """
        self.step += 1

        if self.log_only and not force and self.step % self.logging_freq != 0:
            return

        # ---- ego simulation state ---------------------------------------- #
        ego_t   = self.ego_vehicle.get_transform()
        ego_vel = self.ego_vehicle.get_velocity()
        ego_ext = self.ego_vehicle.bounding_box.extent

        self.ego_location = ego_t.location
        loc = ego_t.location
        ey, ex = ego_ext.y, ego_ext.x

        self.ego_pos    = np.array([[[loc.x, loc.y]]])
        self.ego_yaw    = np.array([[[np.radians(ego_t.rotation.yaw)]]])
        self.ego_vel    = np.array([[[ego_vel.x, ego_vel.y]]])
        self.ego_extent = np.array([[[[ey, ex], [ey, -ex], [-ey, -ex], [-ey, ex]]]])
        self.ego_id     = np.array([[[self.ego_vehicle.id]]])
        self.ego_type   = np.array([[[self.ego_vehicle.type_id]]])
        self.ego_color  = np.array([[[self.ego_vehicle.attributes["color"]]]])

        # ---- background agents ------------------------------------------- #
        if not self.route_only:
            self._initialize_bg_agents()
        self.fetch_bg_state()

        # ---- assemble vehicle state dict ---------------------------------- #
        if self.bg_vehicles and self.bg_color is not None:
            state = {
                "pos":    np.concatenate([self.ego_pos,    self.bg_pos],    axis=1).tolist(),
                "yaw":    np.concatenate([self.ego_yaw,    self.bg_yaw],    axis=1).tolist(),
                "vel":    np.concatenate([self.ego_vel,    self.bg_vel],    axis=1).tolist(),
                "extent": np.concatenate([self.ego_extent, self.bg_extent], axis=1).tolist(),
                "id":     np.concatenate([self.ego_id,     self.bg_id],     axis=1).tolist(),
                "type":   np.concatenate([self.ego_type,   self.bg_type],   axis=1).tolist(),
                "color":  np.concatenate([self.ego_color,  self.bg_color],  axis=1).tolist(),
            }
        else:
            state = {
                "pos":    self.ego_pos.tolist(),
                "yaw":    self.ego_yaw.tolist(),
                "vel":    self.ego_vel.tolist(),
                "extent": self.ego_extent.tolist(),
                "id":     self.ego_id.tolist(),
                "type":   self.ego_type.tolist(),
                "color":  self.ego_color.tolist(),
            }

        # ---- traffic lights ----------------------------------------------- #
        if self.tlights:
            lights = {
                "pos":    self.tl_pos.tolist(),
                "yaw":    self.tl_yaw.tolist(),
                "state":  self.tl_state.tolist(),
                "extent": self.tl_extent.tolist(),
            }
        else:
            lights = {"pos": [], "yaw": [], "state": [], "extent": []}

        # ---- route boxes -------------------------------------------------- #
        # Extract 2D route from planner_state; results are cached and only
        # recomputed every _ROUTE_CACHE_STRIDE route-index steps.
        if planner_state is not None and planner_state.route_points is not None:
            ri = planner_state.route_index
            pts = planner_state.route_points[ri:]
            route_2d = pts[:, :2] if pts.ndim == 2 and pts.shape[1] >= 2 else pts.reshape(-1, 2)
            if len(route_2d) >= 2:
                self.route_as_boxes(route_2d, current_ri=ri)
        if self.route_pos is not None:
            route_boxes = {
                "pos":    self.route_pos.tolist(),
                "yaw":    self.route_yaw.tolist(),
                "id":     self.route_id.tolist(),
                "extent": self.route_extent.tolist(),
            }
        else:
            route_boxes = {"pos": [], "yaw": [], "id": [], "extent": []}

        # ---- ego control -------------------------------------------------- #
        ego_actions = {}
        if ego_control is not None:
            ego_actions = {
                "steer":    ego_control.steer,
                "throttle": ego_control.throttle,
                "brake":    ego_control.brake,
            }

        # ---- write per-frame JSON.gz ------------------------------------- #
        if not self.route_only and (force or self.step % self.logging_freq == 0):
            os.makedirs(self.frames_dir, exist_ok=True)

            frame_data = {
                "meta": {
                    "frame_index":  self.frame_count,
                    "step":         self.step,
                    "route_index":  self.route_index,
                },
                "state":                state,
                "lights":               lights,
                "route_boxes":          route_boxes,
                "ego_actions":          ego_actions,
                "planner_state":        self._serialize_planner_state(planner_state),
                "scene_context":        self._serialize_scene_context(scene_context),
                "prediction_data":      self._serialize_prediction_data(prediction_data),
                "long_planner_result":  self._serialize_long_planner_result(long_planner_result),
                "lat_planner_result":   self._serialize_lat_planner_result(lat_planner_result),
                "ego_plan":             self._serialize_ego_plan(ego_plan),
                "high_level_beh":       self._serialize_high_level_beh(high_level_beh),
                "plan_with_reasoning":  bool(plan_with_reasoning),
            }

            frame_path = os.path.join(self.frames_dir, f"{self.frame_count:06d}.json.gz")
            with gzip.open(frame_path, "wt", encoding="utf-8") as f:
                json.dump(frame_data, f, cls=_NumpyEncoder)

            self.frame_count += 1

        # ---- reset per-step working buffers ------------------------------ #
        # Note: route_pos/yaw/id/extent are intentionally NOT reset — they
        # are cached across steps and reused until the route advances enough.
        self.bg_pos = self.bg_yaw = self.bg_vel = self.bg_extent = None
        self.bg_id  = self.bg_type = self.bg_color = None

        self.tl_pos = self.tl_yaw = self.tl_state = self.tl_extent = None

        return state, lights, route_boxes

    # ---------------------------------------------------------------------- #
    # Persistence
    # ---------------------------------------------------------------------- #

    def dump_to_json(self):
        """Write route metadata for this logging session."""
        if self.route_only:
            return

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        town_name = self.world.get_map().name if self.world is not None else "Unknown"

        meta_data = {
            "index":        self.route_index,
            "town":         town_name,
            "logging_freq": self.logging_freq,
            "frame_count":  self.frame_count,
        }

        meta_path = os.path.join(str(self.save_path), "lon_log_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f)