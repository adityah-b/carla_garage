import numpy as np
import carla

from typing import List, Dict, Optional
from dataclasses import dataclass

from privileged_route_planner import PlannerState
from agents.navigation.global_route_planner import GlobalRoutePlanner

from .lane_handler import LaneHandler, Lanelet

@dataclass(frozen=True, slots=True)
class LaneVehicles:
    lanelet : Lanelet
    vehicles : List[carla.Vehicle]

class RoadHandler:
    """
    Identifies actors present across the road
    """

    def __init__(self, config, carla_map : carla.Map):
        self.config = config
        self.carla_map = carla_map

        self.grp = GlobalRoutePlanner(self.carla_map, self.config.sampling_resolution)

    def _get_all_lanelets(
        self,
        lane_wps : List[carla.Waypoint],
        max_len : float = 80.0,
        backward : bool = False
    ) -> List[Lanelet]:
        """
        Given a list of waypoints, generate list of corresponding lanelets
        """
        all_lanelets : List[Lanelet] = []
        for wp in lane_wps:
            all_lanelets += LaneHandler.generate_lanelet(wp, self.grp, max_len, backward)

        return all_lanelets

    def _group_lanelets(
        self,
        lanelets: List[Lanelet],
        lane_dict: Dict[str, Optional[carla.Waypoint]]
    ) -> Dict[str, List[Lanelet]]:
        """
        Group lanelets by their corresponding lane names.
        """
        grouped_lanelets = {
            lane_name : [] for lane_name in lane_dict.keys()
        }

        for ll in lanelets:
            for name, lane_key in lane_dict.items():
                if not lane_key:
                    continue
                if lane_key in ll:
                    grouped_lanelets[name].append(ll)
                    break

        return grouped_lanelets

    def _assign_vehicles_to_lanelets(
        self,
        vehicles: List[carla.Vehicle],
        grouped_lanelets: Dict[str, List[Lanelet]],
    ) -> Dict[str, List[LaneVehicles]]:
        """
        Given a flat list of vehicles and a dict lane_name→[Lanelet],
        return lane_name→[LaneVehicles], grouping each vehicle
        into the first lanelet whose sparse keyset contains it.
        """
        # Group vehicles by lane name
        grouped_vehicles : Dict[str, List[LaneVehicles]] = {}
        vehicle_wp_map = {
            v : self.carla_map.get_waypoint(v.get_location())
            for v in vehicles
        }

        remaining = set(vehicles)
        for lane_name, lanelets in grouped_lanelets.items():
            lv_list: List[LaneVehicles] = []
            for ll in lanelets:
                vehicles_on_ll = [v for v in list(remaining) if vehicle_wp_map[v] in ll]

                if vehicles_on_ll:
                    lv_list.append(LaneVehicles(lanelet=ll, vehicles=vehicles_on_ll))
                    remaining.difference_update(vehicles_on_ll)

            grouped_vehicles[lane_name] = lv_list

        return grouped_vehicles

    def _filter_vehicles_by_route(
        self,
        planner_state: PlannerState,
        npc_vehicles: List[carla.Vehicle],
        distance_thresh: float,
        yaw_thresh: float,
        traffic_type: str
    ) -> List[carla.Vehicle]:
        """
        Generic route‐relative filter.

        traffic_type:
          - "ongoing":  vehicles traveling roughly same direction ahead of ego
          - "oncoming": vehicles traveling roughly opposite direction, approaching ego
          - "crossing": vehicles traveling roughly perpendicular to ego, crossing its path

        Returns only those vehicles satisfying:
          • min_route_distance < distance_thresh
          AND
          • if leading:   route_yaw_diff < yaw_thresh AND loc_dot >= 0
            if trailing:  test
            if oncoming:  heading_dot < 0      AND loc_dot >= 0
            if crossing:  |heading_dot| < cross_angle_thresh AND loc_dot >= 0
                          AND lateral_dist < lateral_thresh
        """
        # Unpack planner snapshot
        route_index = planner_state.route_index
        route_waypoints = planner_state.route_waypoints
        route_points = planner_state.route_points
        rotation_angles = planner_state.rotation_angles

        if not npc_vehicles or route_index >= route_points.shape[0]:
            return []

        if traffic_type in ["leading", "oncoming", "crossing"]:
            max_detection_radius = self.config.leading_vehicles_maximum_detection_radius
        elif traffic_type == "trailing":
            max_detection_radius = self.config.tailing_vehicles_maximum_detection_radius

        # Slice the route
        if traffic_type in ["leading", "oncoming", "crossing"]:
            route_xy = route_points[route_index : route_index + max_detection_radius, :2][::self.config.points_per_meter]
            route_yaws = rotation_angles[route_index : route_index + max_detection_radius][::self.config.points_per_meter]
        else:
            from_idx = max(0, route_index - max_detection_radius)
            route_xy = route_points[from_idx : route_index, :2][::self.config.points_per_meter]
            route_yaws = rotation_angles[from_idx : route_index][::self.config.points_per_meter]

        # Assemble NPC vehicle arrays
        vehicle_ids = np.array([vehicle.id for vehicle in npc_vehicles])
        vehicle_locs = np.array([[vehicle.get_location().x, vehicle.get_location().y] for vehicle in npc_vehicles])
        vehicle_yaws = np.array([vehicle.get_transform().rotation.yaw for vehicle in npc_vehicles])

        # Calculate NPC vehicle distances to route points
        rel_pos = vehicle_locs[:, None, :] - route_xy[None, :, :]
        rel_dists = np.linalg.norm(rel_pos, axis=2)
        min_route_idxs = rel_dists.argmin(axis=1)
        min_dists = rel_dists[np.arange(len(min_route_idxs)), min_route_idxs]

        # Calculate NPC vehicle yaw differences to route yaws
        min_route_yaws = route_yaws[min_route_idxs]
        yaw_diffs = np.abs((min_route_yaws - vehicle_yaws) % 360)
        yaw_diffs = np.minimum(yaw_diffs, 360 - yaw_diffs)

        # In-front dot-product
        ego_wp = route_waypoints[route_index]
        ego_xy = np.array(
            [ego_wp.transform.location.x, ego_wp.transform.location.y]
        )
        fwd_vec = ego_wp.transform.get_forward_vector()
        ego_fwd_vec = np.array([fwd_vec.x, fwd_vec.y])
        rel_pos_ego_xy = vehicle_locs - ego_xy
        loc_dots = rel_pos_ego_xy @ ego_fwd_vec

        # Heading dot-product (for oncoming and crossing)
        veh_fwd_vecs = np.array([
            [v.get_transform().get_forward_vector().x, v.get_transform().get_forward_vector().y]
            for v in npc_vehicles
        ])
        heading_dots = veh_fwd_vecs @ ego_fwd_vec

        # Build mask
        mask = (min_dists < distance_thresh)
        if traffic_type == "leading":
            mask &= (yaw_diffs < yaw_thresh) & (loc_dots >= 0)
        elif traffic_type == "trailing":
            mask &= (yaw_diffs < yaw_thresh) & (loc_dots < 0)
        elif traffic_type == "oncoming":
            mask &= (heading_dots < 0) & (loc_dots >= 0)
        elif traffic_type == "crossing":
            cross_angle_thresh  = 30  # e.g. ~30°
            # lateral_thresh      = cfg.cross_vehicles_max_lateral_distance  # e.g. road width
            mask &= (np.abs(heading_dots) < np.cos(np.deg2rad(90 - cross_angle_thresh))) \
                    & (loc_dots >= 0)
        else:
            raise ValueError(f"Unknown traffic_type: {traffic_type}")

        # 10) return filtered list
        return [v for v,m in zip(npc_vehicles, mask) if m]

    def get_leading_vehicles(
        self,
        planner_state : PlannerState,
        npc_vehicles : List[carla.Vehicle],
    ) -> Dict[str, List[LaneVehicles]]:
        """
        Get the instances of vehicles leading ahead of the ego vehicle.
        """
        route_index = planner_state.route_index
        route_waypoints = planner_state.route_waypoints
        leading_max_detection_radius = self.config.leading_vehicles_maximum_detection_radius

        # Get the current ego waypoint
        ego_wp = route_waypoints[route_index]
        left_wp = None
        right_wp = None

        # Get the lanes in the same direction as the ego vehicle
        same_lane_wps = LaneHandler.get_same_dir_lanes(ego_wp)

        # Setup the source waypoints for each lanelet
        ego_idx = same_lane_wps.index(ego_wp)
        lane_dict = {'ego' : ego_wp}
        if ego_idx > 0:
            left_wp = same_lane_wps[ego_idx - 1]
            lane_dict["left"] = left_wp
        if ego_idx < len(same_lane_wps) - 1:
            right_wp = same_lane_wps[ego_idx + 1]
            lane_dict["right"] = right_wp

        min_lane_id = left_wp.lane_id if left_wp else ego_wp.lane_id
        max_lane_id = right_wp.lane_id if right_wp else ego_wp.lane_id

        # Define the maximum distance and yaw difference thresholds
        # Get the maximum lane offset from the ego vehicle's lane id
        max_lane_offset = max(abs(min_lane_id - ego_wp.lane_id), abs(max_lane_id - ego_wp.lane_id))
        max_distance = self.config.leading_vehicles_max_route_distance * (1 + max_lane_offset)
        max_yaw_difference = self.config.leading_vehicles_max_route_angle_ongoing

        # Filter all leading vehicles
        leading_vehicles = self._filter_vehicles_by_route(
            planner_state,
            npc_vehicles,
            max_distance,
            max_yaw_difference,
            traffic_type="leading"
        )

        # Generate all lanelets for each target lane waypoint
        all_lanelets = self._get_all_lanelets(same_lane_wps)

        # Group lanelets by their corresponding lanes
        grouped_lanelets = self._group_lanelets(all_lanelets, lane_dict)

        # Group vehicles by lane name
        grouped_vehicles = self._assign_vehicles_to_lanelets(leading_vehicles, grouped_lanelets)

        return grouped_vehicles

    def get_trailing_vehicles(
        self,
        planner_state : PlannerState,
        npc_vehicles : List[carla.Vehicle],
    ) -> Dict[str, List[LaneVehicles]]:
        """
        Get the instances of vehicles trailing behind the ego vehicle.
        """
        route_index = planner_state.route_index
        route_waypoints = planner_state.route_waypoints
        trailing_max_detection_radius = self.config.tailing_vehicles_maximum_detection_radius

        # Get the current ego waypoint
        ego_wp = route_waypoints[route_index]
        left_wp = None
        right_wp = None

        # Get the lanes in the same direction as the ego vehicle
        same_lane_wps = LaneHandler.get_same_dir_lanes(ego_wp)

        # Setup the source waypoints for each lanelet
        ego_idx = same_lane_wps.index(ego_wp)
        lane_dict = {'ego' : ego_wp}
        if ego_idx > 0:
            left_wp = same_lane_wps[ego_idx - 1]
            lane_dict["left"] = left_wp
        if ego_idx < len(same_lane_wps) - 1:
            right_wp = same_lane_wps[ego_idx + 1]
            lane_dict["right"] = right_wp

        min_lane_id = left_wp.lane_id if left_wp else ego_wp.lane_id
        max_lane_id = right_wp.lane_id if right_wp else ego_wp.lane_id

        # Define the maximum distance and yaw difference thresholds
        # Get the maximum lane offset from the ego vehicle's lane id
        max_lane_offset = max(abs(min_lane_id - ego_wp.lane_id), abs(max_lane_id - ego_wp.lane_id))
        max_distance = self.config.trailing_vehicles_max_route_distance * (1 + max_lane_offset)
        max_yaw_difference = self.config.trailing_vehicles_max_route_angle_ongoing

        # Filter all trailing vehicles
        trailing_vehicles = self._filter_vehicles_by_route(
            planner_state,
            npc_vehicles,
            max_distance,
            max_yaw_difference,
            traffic_type="trailing"
        )

        # Generate all lanelets for each target lane waypoint
        all_lanelets = self._get_all_lanelets(same_lane_wps, backward=True)

        # Group lanelets by their corresponding lanes
        grouped_lanelets = self._group_lanelets(all_lanelets, lane_dict)

        # Group vehicles by lane name
        grouped_vehicles = self._assign_vehicles_to_lanelets(trailing_vehicles, grouped_lanelets)

        return grouped_vehicles

    def get_oncoming_vehicles(
        self,
        planner_state : PlannerState,
        npc_vehicles : List[carla.Vehicle],
    ) -> Dict[str, List[LaneVehicles]]:
        """
        Get the instances of vehicles in oncoming traffic with respect to the ego vehicle
        """
        grouped_vehicles = {}
        route_index = planner_state.route_index
        route_waypoints = planner_state.route_waypoints
        leading_max_detection_radius = self.config.leading_vehicles_maximum_detection_radius

        # Get the current ego waypoint
        ego_wp = route_waypoints[route_index]

        # Get the lanes in the opposite direction as the ego vehicle
        opp_lane_wps = LaneHandler.get_opposite_dir_lanes(ego_wp)

        lane_dict = {}
        if opp_lane_wps:
            # Setup the source waypoints for each lanelet
            for i, wp in enumerate(opp_lane_wps):
                lane_dict[f'oncoming-{i}'] = wp

            min_lane_id = opp_lane_wps[-1].lane_id
            max_lane_id = opp_lane_wps[0].lane_id

            # Define the maximum distance and yaw difference thresholds
            # Get the maximum lane offset from the ego vehicle's lane id
            max_lane_offset = max(abs(min_lane_id - ego_wp.lane_id), abs(max_lane_id - ego_wp.lane_id))
            max_distance = self.config.trailing_vehicles_max_route_distance_lane_change * 4
            max_yaw_difference = self.config.leading_vehicles_max_route_angle_oncoming

            # Filter all oncoming vehicles
            oncoming_vehicles = self._filter_vehicles_by_route(
                planner_state,
                npc_vehicles,
                max_distance,
                max_yaw_difference,
                traffic_type="oncoming"
            )

            # Generate all lanelets for each target lane waypoint
            all_lanelets = self._get_all_lanelets(opp_lane_wps, backward=True)

            # Group lanelets by their corresponding lanes
            grouped_lanelets = self._group_lanelets(all_lanelets, lane_dict)

            # Group vehicles by lane name
            grouped_vehicles = self._assign_vehicles_to_lanelets(oncoming_vehicles, grouped_lanelets)

        return grouped_vehicles

    def get_cross_vehicles(
        self,
        planner_state : PlannerState,
        npc_vehicles : List[carla.Vehicle],
    ) -> Dict[str, List[LaneVehicles]]:
        """
        Get the instances of vehicles in cross traffic with respect to the ego vehicle
        """
        grouped_vehicles = {}
        route_index = planner_state.route_index
        route_waypoints = planner_state.route_waypoints
        leading_max_detection_radius = self.config.leading_vehicles_maximum_detection_radius

        # Get the current ego waypoint
        ego_wp = route_waypoints[route_index]

        # Get the lanes in the perpendicular direction as the ego vehicle
        cross_lane_wps = LaneHandler.get_cross_dir_lanes(ego_wp)

        lane_dict = {}
        if cross_lane_wps:
            # Setup the source waypoints for each lanelet
            for wp in cross_lane_wps:
                lane_dict[f'crossing-{wp.lane_id}'] = wp

            # Define the maximum distance and yaw difference thresholds
            max_distance = self.config.trailing_vehicles_max_route_distance_lane_change * 4
            max_yaw_difference = self.config.leading_vehicles_max_route_angle_oncoming

            # Filter all cross vehicles
            cross_vehicles = self._filter_vehicles_by_route(
                planner_state,
                npc_vehicles,
                max_distance,
                max_yaw_difference,
                traffic_type="crossing"
            )

            # Generate all lanelets for each target lane waypoint
            all_lanelets = self._get_all_lanelets(cross_lane_wps, backward=True)

            # Group lanelets by their corresponding lanes
            grouped_lanelets = self._group_lanelets(all_lanelets, lane_dict)

            # Group vehicles by lane name
            grouped_vehicles = self._assign_vehicles_to_lanelets(cross_vehicles, grouped_lanelets)

        return grouped_vehicles
