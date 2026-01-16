import carla
import numpy as np

from typing import List, Dict, Any, Optional, Literal, Tuple
from dataclasses import dataclass
from enum import Enum

from config import GlobalConfig
from privileged_route_planner import PlannerState
from agents.navigation.local_planner import RoadOption
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

class IntersectionType(Enum):
    SIGNALIZED = 0
    UNSIGNALIZED = 1
    OTHER = 2

LaneChangeEntry = Tuple[RoadOption, int, int]
IntersectionEntry = Tuple[RoadOption, int, int, IntersectionType]

@dataclass(frozen=True, slots=True)
class LaneChangeData:
    distance_to_lane_change : float
    inside_lane_change : bool
    target_maneuver: Literal[RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]
    is_executing_maneuver : bool
    entry : LaneChangeEntry


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
    can_change_left : bool
    can_change_right : bool

    @property
    def same_direction_lane_change_available(self) -> bool:
        return self.can_change_left or self.can_change_right

@dataclass(frozen=True, slots=True)
class RouteData:
    lane_info : LaneInfo
    intersection_data : Optional[IntersectionData]
    lane_change_data : Optional[LaneChangeData]

    # highway merge

# TODO: Store intersection and lane change state so as to avoid recomputing the same shit
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
        can_change_left = (
            left_available
            and not left_oncoming
            and (lane_change in (carla.LaneChange.Left, carla.LaneChange.Both) or ego_wp.lane_type != carla.LaneType.Driving)
        )
        can_change_right = (
            right_available
            and not right_oncoming
            and (lane_change in (carla.LaneChange.Right, carla.LaneChange.Both) or ego_wp.lane_type != carla.LaneType.Driving)
        )

        return LaneInfo(
            has_left_lane=left_available,
            has_right_lane=right_available,
            left_oncoming=left_oncoming,
            right_oncoming=right_oncoming,
            can_change_left=can_change_left,
            can_change_right=can_change_right
        )

    def _extract_intersection_data(
        self,
        ego_wp : carla.Waypoint,
        planner_state : PlannerState
    ) -> Optional[IntersectionData]:
        route_index = planner_state.route_index
        ego_tf = ego_wp.transform
        ego_loc = ego_tf.location
        ego_cmd = planner_state.route_commands[route_index]

        intersection_data_raw = self._get_upcoming_intersection(planner_state)
        if not intersection_data_raw:
            return None

        # Unpack intersection information
        intersection_cmd, start_idx, end_idx, signalized = intersection_data_raw

        start_wp = planner_state.route_waypoints[start_idx]
        end_wp = planner_state.route_waypoints[end_idx]

        # passed_start_wp = self._has_passed_waypoint(ego_tf, start_wp)
        # passed_end_wp = self._has_passed_waypoint(ego_tf, end_wp)
        passed_start_wp = route_index >= start_idx
        passed_end_wp = route_index > end_idx

        inside_intersection = (passed_start_wp) and not passed_end_wp

        if not passed_start_wp:
            dist_to_intersection = ego_loc.distance(start_wp.transform.location)
        else:
            dist_to_intersection = 0.0

        is_executing_maneuver = (intersection_cmd == ego_cmd) and inside_intersection

        return IntersectionData(
            distance_to_intersection=dist_to_intersection,
            inside_intersection=inside_intersection,
            signalized=signalized,
            target_maneuver=intersection_cmd,
            is_executing_maneuver=is_executing_maneuver,
            entry=intersection_data_raw
        )

    def _extract_lane_change_data(
        self,
        ego_vehicle : carla.Vehicle,
        planner_state : PlannerState
    ) -> Optional[LaneChangeData]:
        # Get ego state
        ego_tf = ego_vehicle.get_transform()
        ego_loc = ego_tf.location
        ego_speed = ego_vehicle.get_velocity().length()
        ego_cmd = planner_state.route_commands[planner_state.route_index]

        lane_change_data_raw = self._get_upcoming_lane_change(ego_speed, planner_state)
        if not lane_change_data_raw:
            return None

        # Unpack lane change information
        lane_change_cmd, lc_start_idx, lc_end_idx = lane_change_data_raw

        start_wp = planner_state.route_waypoints[lc_start_idx]
        end_wp = planner_state.route_waypoints[lc_end_idx]
        passed_start_wp = self._has_passed_waypoint(ego_tf, start_wp)
        passed_end_wp = self._has_passed_waypoint(ego_tf, end_wp)

        inside_lane_change = (passed_start_wp) and not passed_end_wp

        if not passed_start_wp:
            dist_to_lane_change = ego_loc.distance(start_wp.transform.location)
        else:
            dist_to_lane_change = 0.0

        is_executing_maneuver = (lane_change_cmd == ego_cmd) and inside_lane_change

        return LaneChangeData(
            distance_to_lane_change=dist_to_lane_change,
            inside_lane_change=inside_lane_change,
            target_maneuver=lane_change_cmd,
            is_executing_maneuver=is_executing_maneuver,
            entry=lane_change_data_raw
        )

    def _get_upcoming_intersection(
        self,
        planner_state : PlannerState
    ) -> Optional[IntersectionEntry]:
        # Unpack planner state
        route_index = planner_state.route_index
        route_wps = planner_state.route_waypoints
        route_cmds = planner_state.route_commands

        max_route_length = len(route_wps)
        look_ahead_points = int(self.LOOKAHEAD_DISTANCE * self.config.points_per_meter)
        to_index = min(max_route_length - 1, route_index + look_ahead_points)

        # Look for upcoming intersection
        intersection_start_idx = None
        intersection_cmd = None
        for i in range(route_index, to_index):
            cmd = route_cmds[i]
            wp = route_wps[i]
            if (wp.is_junction) and (cmd in (RoadOption.LEFT, RoadOption.STRAIGHT, RoadOption.LANEFOLLOW, RoadOption.RIGHT)):
                intersection_start_idx = i
                intersection_cmd = cmd
                break

        if intersection_start_idx is None:
            return None

        intersection_end_idx = intersection_start_idx
        while (intersection_end_idx < max_route_length - 1) and route_cmds[intersection_end_idx] == intersection_cmd:
            intersection_end_idx +=1

        print(f'\n\nINTERSECTION TURN')
        print(f'\t\tstart: {intersection_start_idx}, end: {intersection_end_idx}')

        # Any intersections with traffic lights are automatically signalized
        dist_next_tl = planner_state.dist_to_next_traffic_lights[route_index]

        dist_to_junc = route_wps[intersection_start_idx].transform.location.distance(route_wps[route_index].transform.location)

        # TODO: IDEALLY ALL THIS SHOULD BE PRECOMPUTED IN THE PRIVILEGEDROUTEPLANNER
        if dist_next_tl != np.inf and abs(dist_next_tl - dist_to_junc) < 10.0:
            return (intersection_cmd, intersection_start_idx, intersection_end_idx, IntersectionType.SIGNALIZED)

        return (intersection_cmd, intersection_start_idx, intersection_end_idx, IntersectionType.UNSIGNALIZED)

    def _get_upcoming_lane_change(
        self,
        ego_speed : float,
        planner_state : PlannerState
    ) -> Optional[LaneChangeEntry]:
        # Unpack planner state
        route_index = planner_state.route_index
        route_pts = planner_state.route_points
        route_cmds = planner_state.route_commands

        # Calculate the braking distance based on the ego speed
        braking_distance = ((
            (ego_speed * 3.6) / 10.0)**2 / 2.0) + self.config.braking_distance_calculation_safety_distance

        # Determine the number of waypoints to look ahead based on the braking distance
        look_ahead_points = max(self.LOOKAHEAD_DISTANCE * self.config.points_per_meter,
            min(route_pts.shape[0], self.config.points_per_meter * int(braking_distance)))
        look_ahead_points = int(look_ahead_points)
        max_route_length = len(route_cmds)

        to_index = min(max_route_length - 1, route_index + look_ahead_points)

        # Iterate over the points around the current position, checking for lane change commands
        lc_start_idx = None
        lane_change_cmd = None
        for i in range(route_index, to_index):
            cmd = route_cmds[i]
            if cmd in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
                # Set the lane change direction and mandatory start point to begin the maneuver
                lc_start_idx = i
                lane_change_cmd = cmd
                break

        if lc_start_idx is None:
            return None

        # Find the end point of the lane change, where the lane change is completed
        lc_end_idx = lc_start_idx
        while (lc_end_idx < max_route_length - 1) and route_cmds[lc_end_idx] == lane_change_cmd:
            lc_end_idx += 1

        print(f'\n\nLANE CHANGE')
        print(f'\t\tstart: {lc_start_idx}, end: {lc_end_idx}')

        return (lane_change_cmd, lc_start_idx, lc_end_idx)