import numpy as np
import carla

from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import OrderedDict

from .junction_handler import JunctionHandler
from agents.navigation.global_route_planner import GlobalRoutePlanner

@dataclass(frozen=True, slots=True)
class Lanelet:
    points : OrderedDict[tuple[int, int, int], carla.Waypoint]

    def waypoints_list(self):
        return list(self.points.values())

    def lanelet_sections(self):
        return set(self.points.keys())

    def __len__(self):
        return len(self.points)

    def __contains__(self, wp: carla.Waypoint) -> bool:
        key = (wp.road_id, wp.section_id, wp.lane_id)
        return key in self.points

class WaypointUtils:
    WAYPOINT_SAMPLING_RESOLUTION : float = 2.0

    @staticmethod
    def _find_waypoint_entry_exit(
        grp : GlobalRoutePlanner,
        wp : carla.Waypoint
    ) -> Tuple[Optional[carla.Waypoint], Optional[carla.Waypoint]]:
        """
        Find the entry and exit points of the waypoint's corresponding road segment.
        Args:
            waypoint (carla.Waypoint): The waypoint.

        Returns:
            tuple: A tuple containing the entry and exit points of the road segment.
        """
        entry_wp, exit_wp = None, None

        try:
            node_pair = grp._road_id_to_edge[wp.road_id][wp.section_id][wp.lane_id]
        except KeyError:
            print(f'Could not find edge for waypoint {wp.transform.location}')

        if node_pair:
            entry_xyz, exit_xyz = node_pair
            edge = grp._graph.edges[entry_xyz, exit_xyz]
            entry_wp = edge['entry_waypoint']
            exit_wp = edge['exit_waypoint']

        return (entry_wp, exit_wp)

    @staticmethod
    def get_distance(
        wp1 : carla.Waypoint,
        wp2 : carla.Waypoint
    ) -> float:
        """
        Get the distance between two waypoints
        """
        return wp1.transform.location.distance(wp2.transform.location)

    @staticmethod
    def interpolate_between_waypoints(
        source_wp : carla.Waypoint,
        target_wp : carla.Waypoint,
        grp : GlobalRoutePlanner,
        assume_no_junction : bool = False
    ) -> List[carla.Waypoint]:
        """
        Interpolate waypoints between source and target waypoints

        Args:
            source_wp (carla.Waypoint): Source waypoint.
            target_wp (carla.Waypoint): Target waypoint.

        Returns:
            list: List of interpolated waypoints between source and target waypoints up to desired distance.
        """
        # print(f'Interpolating between waypoints {source_wp.transform.location} and {target_wp.transform.location}')
        interp_wps = [source_wp]
        cur_wp = source_wp

        while WaypointUtils.get_distance(cur_wp, target_wp) > WaypointUtils.WAYPOINT_SAMPLING_RESOLUTION and \
                WaypointUtils.get_distance(cur_wp, source_wp) < WaypointUtils.get_distance(target_wp, source_wp):
            wp_choice = cur_wp.next(WaypointUtils.WAYPOINT_SAMPLING_RESOLUTION)
            if not assume_no_junction and len(wp_choice) > 1:
                max_dot = -1 * np.inf
                target_wp_vec = target_wp.transform.get_forward_vector()
                for wp in wp_choice:
                    wp_vec = wp.transform.get_forward_vector()
                    target_select_wp_dot = target_wp_vec.dot(wp_vec)
                    # Select waypoint with straightest path to target waypoint
                    if target_select_wp_dot > max_dot:
                        max_dot = target_select_wp_dot
                        cur_wp = wp

            elif not wp_choice:
                _, exit_wp = WaypointUtils._find_waypoint_entry_exit(grp, cur_wp)
                cur_wp = exit_wp

            else:
                cur_wp = wp_choice[0]

            interp_wps.append(cur_wp)

        interp_wps.append(target_wp)

        return interp_wps

    @staticmethod
    def get_waypoint_at_distance(
        source_wp : carla.Waypoint,
        grp : GlobalRoutePlanner,
        distance : float,
    ) -> carla.Waypoint:
        """
        Get a waypoint at a certain distance from the input waypoint.

        Args:
            waypoint (carla.Waypoint): The input waypoint.
            distance (float): The distance to travel.
            ego_entry_exit_pairs (list): List of entry and exit points of the ego vehicle's route.

        Returns:
            list: List of waypoints at the specified distance.
        """
        # print(f'Getting waypoints at distance {distance} from {waypoint.transform.location}')
        traveled_distance = 0.0
        cur_wp = source_wp

        while traveled_distance < distance:
            wp_choice = cur_wp.next(WaypointUtils.WAYPOINT_SAMPLING_RESOLUTION)
            if not wp_choice:
                _, exit_wp = WaypointUtils._find_waypoint_entry_exit(grp, cur_wp)
                cur_wp = exit_wp
            else:
                cur_wp = wp_choice[0]

            traveled_distance = WaypointUtils.get_distance(cur_wp, source_wp)

        return cur_wp

class LaneHandler:
    """
    Identifies all ongoing and oncoming lanes
    """

    @staticmethod
    def _to_lanelet(
        waypoints : List[carla.Waypoint]
    ) -> Lanelet:
        """
        Generate lanelet given waypoint list
        """
        lanelet_points = OrderedDict()
        for wp in waypoints:
            key = (wp.road_id, wp.section_id, wp.lane_id)
            lanelet_points.setdefault(key, wp)

        return Lanelet(lanelet_points)

    @staticmethod
    def generate_lanelet(
        start_wp : carla.Waypoint,
        grp : GlobalRoutePlanner,
        max_length : float = 50.0
    ) -> List[Lanelet]:
        """
        Generate lanelets that extend max_length metres from start_wp or through the next junction
        """
        lanelets = []
        # Look ahead for any junctions
        junction_pair = JunctionHandler.get_next_junction(start_wp, max_length)
        if junction_pair:
            junction_entry_wp, junction_wp = junction_pair
            junction_map = JunctionHandler.create_junction_map(junction_wp)
            junction_connections = JunctionHandler.get_junction_connections(junction_map, junction_entry_wp)

            for j_conn in junction_connections:
                anchors = [
                    start_wp,
                    j_conn.entry_connection,
                    j_conn.entry_junction,
                    j_conn.exit_junction,
                    j_conn.exit_connection
                ]

                dense_lanelet = []
                for a, b in zip(anchors[:-1], anchors[1:]):
                    dense_lanelet.extend(WaypointUtils.interpolate_between_waypoints(a, b, grp))

                lanelets.append(LaneHandler._to_lanelet(dense_lanelet))
        else:
            end_wp_choices = start_wp.next(max_length)
            end_wp = end_wp_choices[0] if end_wp_choices else WaypointUtils.get_waypoint_at_distance(start_wp, grp, max_length)
            dense_lanelet = WaypointUtils.interpolate_between_waypoints(start_wp, end_wp, grp)
            lanelets.append(LaneHandler._to_lanelet(dense_lanelet))

        return lanelets

    @staticmethod
    def get_same_dir_lanes(
        waypoint : carla.Waypoint
    ) -> List[carla.Waypoint]:
        """
        Gets immediate left and right lanes with the same direction of the road of a wp.

        Args:
            waypoint (carla.Waypoint): Waypoint to start the search from.

        Returns:
            list: List of waypoints with the same direction of the road.
        """
        same_dir_wps = [waypoint]

        # Check roads on the right
        possible_right_wp = waypoint.get_right_lane()
        if possible_right_wp and possible_right_wp.lane_type == carla.LaneType.Driving:
            same_dir_wps.append(possible_right_wp)

        # Check roads on the left
        possible_left_wp = waypoint.get_left_lane()
        if possible_left_wp and possible_left_wp.lane_type == carla.LaneType.Driving and possible_left_wp.lane_id * waypoint.lane_id >= 0:
            same_dir_wps.insert(0, possible_left_wp)

        return same_dir_wps

    @staticmethod
    def get_opposite_dir_lanes(
        waypoint : carla.Waypoint
    ) -> List[carla.Waypoint]:
        """
        Gets all the lanes with opposite direction of the road of a wp
        Ordered from the center lane to the edge one (from inwards to outwards)

        Args:
            waypoint (carla.Waypoint): Waypoint to start the search from.

        Returns:
            list: List of waypoints with opposite direction of the road.
        """
        other_dir_wps = []
        other_dir_wp = None

        # Get the first lane of the opposite direction
        left_wp = waypoint
        while True:
            possible_left_wp = left_wp.get_left_lane()
            if possible_left_wp is None:
                break
            if possible_left_wp.lane_id * left_wp.lane_id < 0:
                other_dir_wp = possible_left_wp
                break
            left_wp = possible_left_wp

        if not other_dir_wp:
            return other_dir_wps

        # Check roads on the right
        right_wp = other_dir_wp
        while True:
            if right_wp.lane_type == carla.LaneType.Driving:
                other_dir_wps.append(right_wp)
            possible_right_wp = right_wp.get_right_lane()
            if possible_right_wp is None:
                break
            right_wp = possible_right_wp

        return other_dir_wps

    # TODO: Implement cross direction checking
    @staticmethod
    def get_cross_dir_lanes(
        waypoint : carla.Waypoint
    ) -> List[carla.Waypoint]:
        pass
