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

LaneChangeEntry = Tuple[RoadOption, int, int, int]
IntersectionEntry = Tuple[RoadOption, int, int, IntersectionType]

@dataclass(frozen=True, slots=True)
class LaneChangeData:
    distance_to_lane_change : float
    inside_lane_change : bool
    available_lane_change_distance: float
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
class ObstacleData:
    distance_to_obstacle : float

@dataclass(frozen=True, slots=True)
class RouteData:
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
        return RouteData(
            intersection_data=self._extract_intersection_data(ego_wp, planner_state),
            lane_change_data=self._extract_lane_change_data(ego_vehicle, planner_state)
        )

    def _extract_intersection_data(
        self,
        ego_wp : carla.Waypoint,
        planner_state : PlannerState
    ) -> Optional[IntersectionData]:
        ego_tf = ego_wp.transform
        ego_loc = ego_tf.location
        ego_cmd = planner_state.route_commands[planner_state.route_index]

        intersection_data_raw = self._get_upcoming_intersection(planner_state)
        if not intersection_data_raw:
            return None

        # Unpack intersection information
        intersection_cmd, start_idx, end_idx, signalized = intersection_data_raw

        start_wp = planner_state.route_waypoints[start_idx]
        end_wp = planner_state.route_waypoints[end_idx]

        passed_start_wp = self._has_passed_waypoint(ego_tf, start_wp)
        passed_end_wp = self._has_passed_waypoint(ego_tf, end_wp)

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
        lane_change_cmd, early_start_idx, late_start_idx, end_idx = lane_change_data_raw

        early_start_wp = planner_state.route_waypoints[early_start_idx]
        late_start_wp = planner_state.route_waypoints[late_start_idx]

        passed_early_wp = self._has_passed_waypoint(ego_tf, early_start_wp)
        passed_late_wp = self._has_passed_waypoint(ego_tf, late_start_wp)

        inside_lane_change = (passed_early_wp) and not passed_late_wp

        if not passed_early_wp:
            dist_to_lane_change = ego_loc.distance(early_start_wp.transform.location)
        else:
            dist_to_lane_change = 0.0

        avail_dist = 0.0 if passed_late_wp else ego_loc.distance(late_start_wp.transform.location)

        is_executing_maneuver = (lane_change_cmd == ego_cmd) and inside_lane_change

        return LaneChangeData(
            distance_to_lane_change=dist_to_lane_change,
            inside_lane_change=inside_lane_change,
            available_lane_change_distance=avail_dist,
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
        next_tl = planner_state.next_traffic_lights[route_index]
        next_ss = planner_state.next_stop_signs[route_index]

        max_route_length = len(route_wps)
        look_ahead_points = int(self.LOOKAHEAD_DISTANCE * self.config.points_per_meter)
        to_index = min(max_route_length - 1, route_index + look_ahead_points)

        # Look for upcoming intersection
        intersection_idx = None
        intersection_cmd = None
        for i in range(route_index, to_index):
            cmd = route_cmds[i]
            wp = route_wps[i]
            if (wp.is_junction) and (cmd in (RoadOption.LEFT, RoadOption.STRAIGHT, RoadOption.RIGHT)):
                intersection_idx = i
                intersection_cmd = cmd
                break

        if intersection_idx is None:
            return None

        end_idx = intersection_idx
        while (end_idx < max_route_length) and route_wps[end_idx].is_junction and route_cmds[end_idx] == intersection_cmd:
            end_idx +=1

        if next_tl and not next_ss:
            signalized = IntersectionType.SIGNALIZED
        elif not next_tl and next_ss:
            signalized = IntersectionType.UNSIGNALIZED
        else:
            signalized = IntersectionType.OTHER

        return (intersection_cmd, intersection_idx, end_idx, signalized)

    def _get_upcoming_lane_change(
        self,
        ego_speed : float,
        planner_state : PlannerState
    ) -> Optional[LaneChangeEntry]:
        # Unpack planner state
        route_index = planner_state.route_index
        route_pts = planner_state.route_points
        route_wps = planner_state.route_waypoints
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
        lane_change_idx = None
        lane_change_cmd = None
        for i in range(route_index, to_index):
            cmd = route_cmds[i]
            if cmd in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
                # Set the lane change direction and mandatory start point to begin the maneuver
                lane_change_idx = i
                lane_change_cmd = cmd
                break

        if lane_change_idx is None:
            return None

        late_wp = route_wps[lane_change_idx]

        early_idx = lane_change_idx
        traveled_distance = 0.0

        while (early_idx > route_index) and \
            (route_wps[early_idx].lane_id == late_wp.lane_id) and \
            (route_wps[early_idx].road_id == late_wp.road_id) and \
            (traveled_distance < self.LOOKAHEAD_DISTANCE):

            early_idx -= 1
            traveled_distance = late_wp.transform.location.distance(route_wps[early_idx].transform.location)

        print(f'R IDX: {route_index}, E IDX: {early_idx}, LC IDX: {lane_change_idx}')
        print(f'Traveled distance: {traveled_distance}')

        # Find the end point of the lane change, where the lane change is completed
        end_idx = lane_change_idx
        while (end_idx < max_route_length) and route_cmds[end_idx] == lane_change_cmd:
            end_idx += 1

        return (lane_change_cmd, early_idx, lane_change_idx, end_idx)