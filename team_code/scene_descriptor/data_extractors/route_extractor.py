import carla
import numpy as np

from typing import List, Dict, Any, Optional, Literal, Tuple
from dataclasses import dataclass

from config import GlobalConfig
from privileged_route_planner import PlannerState, IntersectionType, IntersectionSegment, LaneChangeSegment
from agents.navigation.local_planner import RoadOption
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

LaneChangeEntry = Tuple[RoadOption, int, int]
IntersectionEntry = Tuple[RoadOption, int, int, IntersectionType]

@dataclass(frozen=True, slots=True)
class LaneChangeData:
    distance_to_lane_change : float
    inside_lane_change : bool
    target_maneuver: Literal[RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]
    is_executing_maneuver : bool
    entry : LaneChangeEntry
    lane_boundary_idx : int

@dataclass(frozen=True, slots=True)
class IntersectionData:
    distance_to_intersection : float
    inside_intersection : bool
    signalized : IntersectionType
    target_maneuver : Literal[RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT]
    is_executing_maneuver : bool
    entry : IntersectionEntry

@dataclass(frozen=True, slots=True)
class LaneInfo:
    has_left_lane : bool
    has_right_lane : bool

    left_oncoming : bool
    right_oncoming : bool

    left_same_dir : bool
    right_same_dir : bool

    left_wp : Optional[carla.Waypoint]
    right_wp : Optional[carla.Waypoint]

    @property
    def same_direction_lane_change_available(self) -> bool:
        return self.left_same_dir or self.right_same_dir

@dataclass(frozen=True, slots=True)
class RouteData:
    lane_info : LaneInfo
    intersection_data : Optional[IntersectionData]
    lane_change_data : Optional[LaneChangeData]

    # highway merge

class RouteDataExtractor:
    LOOKAHEAD_DISTANCE = 50.0

    def __init__(self, config : GlobalConfig):
        self.config = config

    def _has_passed_waypoint(
        self,
        ego_tf : carla.Transform,
        wp : carla.Waypoint
    ) -> bool:
        ego_fwd_vec = ego_tf.get_forward_vector()
        rel_ego_wp = wp.transform.location - ego_tf.location
        return ego_fwd_vec.dot(rel_ego_wp) < 0.0

    def extract_route_data(
        self,
        ego_vehicle : carla.Vehicle,
        ego_wp : carla.Waypoint,
        planner_state : PlannerState
    ) -> RouteData:
        # TODO: TEMP CHANGE FOR TESTING
        lane_info = self._extract_lane_info(ego_wp)
        intersection_data=self._extract_intersection_data(ego_wp, planner_state)
        lane_change_data=self._extract_lane_change_data(ego_vehicle, planner_state)

        if intersection_data and lane_change_data:
            if intersection_data.distance_to_intersection < lane_change_data.distance_to_lane_change:
                lane_change_data = None
            else:
                intersection_data = None

        return RouteData(
            lane_info=lane_info,
            intersection_data=intersection_data,
            lane_change_data=lane_change_data
        )

    def _extract_lane_info(self, ego_wp: carla.Waypoint) -> LaneInfo:
        left_wp = ego_wp.get_left_lane()
        right_wp = ego_wp.get_right_lane()

        left_available = left_wp is not None and left_wp.lane_type == carla.LaneType.Driving
        right_available = right_wp is not None and right_wp.lane_type == carla.LaneType.Driving

        ego_forward = ego_wp.transform.get_forward_vector()

        def _is_oncoming(neighbor_wp: Optional[carla.Waypoint]) -> bool:
            if neighbor_wp is None:
                return False
            neighbor_forward = neighbor_wp.transform.get_forward_vector()
            return ego_forward.dot(neighbor_forward) < 0.0

        left_oncoming = left_available and _is_oncoming(left_wp)
        right_oncoming = right_available and _is_oncoming(right_wp)

        lane_change = ego_wp.lane_change
        left_same_dir = (
            left_available
            and not left_oncoming
            and (lane_change in (carla.LaneChange.Left, carla.LaneChange.Both) or ego_wp.lane_type != carla.LaneType.Driving)
        )
        right_same_dir = (
            right_available
            and not right_oncoming
            and (lane_change in (carla.LaneChange.Right, carla.LaneChange.Both) or ego_wp.lane_type != carla.LaneType.Driving)
        )

        return LaneInfo(
            has_left_lane=left_available,
            has_right_lane=right_available,
            left_oncoming=left_oncoming,
            right_oncoming=right_oncoming,
            left_same_dir=left_same_dir,
            right_same_dir=right_same_dir,
            left_wp=left_wp,
            right_wp=right_wp,
        )

    def _extract_intersection_data(
        self,
        ego_wp : carla.Waypoint,
        planner_state : PlannerState
    ) -> Optional[IntersectionData]:
        segment = self._find_intersection_segment(planner_state)
        if not segment:
            return None

        route_index = planner_state.route_index
        ego_loc     = ego_wp.transform.location
        ego_cmd     = planner_state.route_commands[route_index]

        start_idx        = segment.start_idx
        end_idx          = segment.end_idx
        intersection_cmd = segment.cmd
        # Use the signalization type captured at route-build time so it is stable
        # even after the vehicle has passed the traffic control device.
        signalized = segment.signalized

        start_wp = planner_state.route_waypoints[start_idx]

        passed_start_wp    = route_index >= start_idx
        passed_end_wp      = route_index > end_idx
        inside_intersection = passed_start_wp and not passed_end_wp

        dist_to_intersection  = 0.0 if passed_start_wp else ego_loc.distance(start_wp.transform.location)
        is_executing_maneuver = (intersection_cmd == ego_cmd) and inside_intersection

        entry = (intersection_cmd, start_idx, end_idx, signalized)

        return IntersectionData(
            distance_to_intersection=dist_to_intersection,
            inside_intersection=inside_intersection,
            signalized=signalized,
            target_maneuver=intersection_cmd,
            is_executing_maneuver=is_executing_maneuver,
            entry=entry
        )

    def _extract_lane_change_data(
        self,
        ego_vehicle : carla.Vehicle,
        planner_state : PlannerState
    ) -> Optional[LaneChangeData]:
        ego_tf    = ego_vehicle.get_transform()
        ego_loc   = ego_tf.location
        ego_speed = ego_vehicle.get_velocity().length()
        ego_cmd   = planner_state.route_commands[planner_state.route_index]

        segment = self._find_lane_change_segment(ego_speed, planner_state)
        if not segment:
            return None

        lane_change_cmd = segment.cmd
        lc_start_idx    = segment.start_idx
        lc_end_idx      = segment.end_idx

        start_wp = planner_state.route_waypoints[lc_start_idx]
        end_wp   = planner_state.route_waypoints[lc_end_idx]

        passed_start_wp = self._has_passed_waypoint(ego_tf, start_wp)
        passed_end_wp   = self._has_passed_waypoint(ego_tf, end_wp)

        inside_lane_change  = passed_start_wp and not passed_end_wp
        dist_to_lane_change = 0.0 if passed_start_wp else ego_loc.distance(start_wp.transform.location)
        is_executing_maneuver = (lane_change_cmd == ego_cmd) and inside_lane_change

        entry = (lane_change_cmd, lc_start_idx, lc_end_idx)

        return LaneChangeData(
            distance_to_lane_change=dist_to_lane_change,
            inside_lane_change=inside_lane_change,
            target_maneuver=lane_change_cmd,
            is_executing_maneuver=is_executing_maneuver,
            entry=entry,
            lane_boundary_idx=segment.boundary_idx,
        )

    def _find_intersection_segment(
        self,
        planner_state : PlannerState
    ) -> Optional[IntersectionSegment]:
        """Return the first intersection segment at or ahead of the current route index.

        Reads directly from the precomputed list in PlannerState — no per-tick
        search or mutable cache state required.
        """
        route_index = planner_state.route_index
        for seg in planner_state.intersection_segments:
            if seg.end_idx >= route_index:
                return seg
        return None

    def _find_lane_change_segment(
        self,
        ego_speed : float,
        planner_state : PlannerState
    ) -> Optional[LaneChangeSegment]:
        """Return the first lane change segment within braking/lookahead distance.

        Reads directly from the precomputed list in PlannerState — no per-tick
        search or mutable cache state required.
        """
        route_index = planner_state.route_index

        braking_distance = (
            ((ego_speed * 3.6) / 10.0) ** 2 / 2.0
            + self.config.braking_distance_calculation_safety_distance
        )
        look_ahead_m  = max(self.LOOKAHEAD_DISTANCE, braking_distance)
        look_ahead_pts = int(look_ahead_m * self.config.points_per_meter)
        to_index = route_index + look_ahead_pts

        for seg in planner_state.lane_change_segments:
            if seg.end_idx >= route_index and seg.start_idx <= to_index:
                return seg
        return None