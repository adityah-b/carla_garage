import numpy as np
import carla
import os

from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

from privileged_route_planner import PlannerState

from .lane_handler import LaneHandler, Lanelet
from .lanelet_graph import LaneletGraph

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

        # ws_dir = Path(os.environ['WORK_DIR'])
        # lanelet_dir = ws_dir.joinpath('team_code/scene_descriptor/lanelets')
        # map_name = self.carla_map.name.split('/')[-1]
        # lanelet_file = lanelet_dir / (map_name + '.h5')
        # print(f"\nLoading lanelet graph from {lanelet_file}")

        # self.grp = GlobalRoutePlanner(self.carla_map, self.config.sampling_resolution)
        # self.lanelet_graph = LaneletGraph.load_from_h5(self.carla_map, lanelet_file)

    def _get_all_lanelets(
        self,
        lane_wps : List[carla.Waypoint],
        max_len : float = 80.0,
        backward : bool = False,
        bidirectional : bool = False,
    ) -> List[Lanelet]:
        """
        Given a list of waypoints, generate list of corresponding lanelets
        """
        all_lanelets : List[Lanelet] = []
        for wp in lane_wps:
            all_lanelets += LaneHandler.generate_lanelet(wp, max_len, bidirectional, backward)

        return all_lanelets

    # def _get_all_lanelets(
    #     self,
    #     lane_wps : List[carla.Waypoint],
    #     max_len : float = 80.0,
    #     backward : bool = False
    # ) -> List[Lanelet]:
    #     """
    #     Given a list of waypoints, generate list of corresponding lanelets
    #     """
    #     all_lanelets : List[Lanelet] = []
    #     for wp in lane_wps:
    #         found_lanelet = self.lanelet_graph.find_lanelet(wp)
    #         if found_lanelet:
    #             n1, n2, _ = found_lanelet
    #             all_lanelets += [self.lanelet_graph.get_connected_lanelets(n1, n2, max_len, direction='forward')]

    #     return all_lanelets

    def _group_lanelets(
        self,
        lanelets: List[Lanelet],
        lane_dict: Dict[str, List[carla.Waypoint]]
    ) -> Dict[str, List[Lanelet]]:
        """
        Group lanelets by their corresponding lane names.
        """
        grouped_lanelets = {
            lane_name : [] for lane_name in lane_dict.keys()
        }

        for ll in lanelets:
            for name, lane_keys in lane_dict.items():
                if not lane_keys:
                    continue
                if any((k in ll) for k in lane_keys):
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
            v : self.carla_map.get_waypoint(v.get_location(), lane_type=carla.LaneType.Driving)
            for v in vehicles
        }

        remaining = set(vehicles)
        for lane_name, lanelets in grouped_lanelets.items():
            lv_list: List[LaneVehicles] = []
            for ll in lanelets:
                vehicles_on_ll = []
                for v in remaining:
                    if vehicle_wp_map[v] in ll:
                        vehicles_on_ll.append(v)
                    elif ll.contains_wp(vehicle_wp_map[v]):
                        vehicles_on_ll.append(v)

                if vehicles_on_ll:
                    lv_list.append(LaneVehicles(lanelet=ll, vehicles=vehicles_on_ll))
                    remaining.difference_update(vehicles_on_ll)

            grouped_vehicles[lane_name] = lv_list

        return grouped_vehicles

    def _filter_vehicles_by_route(
        self,
        ego_transform : carla.Transform,
        ego_wp : carla.Waypoint,
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
            route_xy = route_points[from_idx : route_index + 1, :2][::self.config.points_per_meter]
            route_yaws = rotation_angles[from_idx : route_index + 1][::self.config.points_per_meter]

            if route_index - max_detection_radius < 0:
                distance_thresh = max(distance_thresh, (max_detection_radius - route_index) / self.config.points_per_meter)

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
        # ego_wp = route_waypoints[route_index]
        ego_xy = np.array([ego_transform.location.x, ego_transform.location.y])
        fwd_vec = ego_transform.get_forward_vector()
        # print(f'\n\nEGO FWD VEC\n\t\tlen: {fwd_vec.length()}')
        ego_fwd_vec = np.array([fwd_vec.x, fwd_vec.y])
        # print(f'EGO FWD 2D VEC\n\t\tLEN: {np.linalg.norm(ego_fwd_vec)}')
        rel_pos_ego_xy = vehicle_locs - ego_xy
        loc_dots = rel_pos_ego_xy @ ego_fwd_vec

        # Heading dot-product (for oncoming and crossing)
        veh_fwd_vecs = np.array([
            [v.get_transform().get_forward_vector().x, v.get_transform().get_forward_vector().y]
            for v in npc_vehicles
        ])
        heading_dots = veh_fwd_vecs @ ego_fwd_vec

        # Distance to ego filters
        veh_to_ego_dists = np.linalg.norm(rel_pos_ego_xy, axis=1)
        # ego_distance_thresh = self.config.detection_radius
        ego_distance_thresh = 80.0

        # TODO: ENSURE RELEVANT VECTORS ARE NORMALIZED AND CLEAN UP CODE
        # Build mask
        mask = (veh_to_ego_dists < ego_distance_thresh)
        # mask = (min_dists < distance_thresh) & (veh_to_ego_dists < ego_distance_thresh)
        if traffic_type == "leading":
            mask &= (min_dists < distance_thresh) & (yaw_diffs < yaw_thresh) & (loc_dots >= 0)
            # mask &= (np.abs(heading_dots) < np.cos(np.deg2rad(0 - yaw_thresh))) & (yaw_diffs < yaw_thresh) & (loc_dots >= 0)
        elif traffic_type == "trailing":
            trailing_angle_thresh = self.config.trailing_vehicles_max_route_angle_ongoing
            trailing_heading_thresh = np.cos(np.deg2rad(trailing_angle_thresh))
            mask &= (heading_dots >= trailing_heading_thresh) & (loc_dots < 0)
            # mask &= (yaw_diffs < yaw_thresh) & (loc_dots < 0)
        elif traffic_type == "oncoming":
            oncoming_angle = 75 if ego_wp.is_junction else 60
            oncoming_heading_thresh = np.cos(np.deg2rad(oncoming_angle))
            mask &= (heading_dots < -oncoming_heading_thresh)

            if not ego_wp.is_junction:
                mask &= (loc_dots >= 0)
            else:
                front_cone_deg = 130
                front_thresh = np.cos(np.deg2rad(front_cone_deg))
                mask &= (loc_dots / (veh_to_ego_dists + 1e-6) > front_thresh)

            # mask &= (heading_dots < -oncoming_heading_thresh) & (loc_dots >= 0)
        elif traffic_type == "crossing":
            cross_angle_thresh  = 45  # e.g. ~30°
            # lateral_thresh      = cfg.cross_vehicles_max_lateral_distance  # e.g. road width
            mask &= (np.abs(heading_dots) < np.cos(np.deg2rad(90 - cross_angle_thresh)))
            if not ego_wp.is_junction:
                mask &= (loc_dots >= 0)
        else:
            raise ValueError(f"Unknown traffic_type: {traffic_type}")

        # 10) return filtered list
        return [v for v,m in zip(npc_vehicles, mask) if m and not v.get_control().hand_brake]

    def get_leading_vehicles(
        self,
        ego_transform : carla.Transform,
        ego_wp : carla.Waypoint,
        planner_state : PlannerState,
        npc_vehicles : List[carla.Vehicle],
    ) -> Dict[str, List[LaneVehicles]]:
        """
        Get the instances of vehicles leading ahead of the ego vehicle.
        """
        route_waypoints = planner_state.route_waypoints[planner_state.route_index:]

        # Get the lanes in the same direction as the ego vehicle
        lane_info = LaneHandler.get_same_dir_lanes(ego_wp, route_waypoints)

        entry_corridor = lane_info["entry_corridor"]     # left?, ref, right?
        exit_corridor = lane_info["exit_corridor"]
        entry_wp = lane_info["entry_wp"]
        exit_wp = lane_info["exit_wp"]
        seeds = lane_info["seeds"]

        # Setup the source waypoints for each lanelet
        left_entry_wp = right_entry_wp = None
        left_exit_wp = right_exit_wp = None

        ego_entry_idx = entry_corridor.index(entry_wp)
        if ego_entry_idx > 0:
            left_entry_wp = entry_corridor[ego_entry_idx - 1]
        if ego_entry_idx < len(entry_corridor) - 1:
            right_entry_wp = entry_corridor[ego_entry_idx + 1]

        if exit_corridor:
            ego_exit_idx = exit_corridor.index(exit_wp)
            if ego_exit_idx > 0:
                left_exit_wp = exit_corridor[ego_exit_idx - 1]
            if ego_exit_idx < len(exit_corridor) - 1:
                right_exit_wp = exit_corridor[ego_exit_idx + 1]

        lane_dict = {
            "left":  [w for w in [left_entry_wp, left_exit_wp] if w is not None],
            "ego":   [w for w in [entry_wp, exit_wp] if w is not None],
            "right": [w for w in [right_entry_wp, right_exit_wp] if w is not None],
        }

        min_lane_id = left_entry_wp.lane_id if left_entry_wp else ego_wp.lane_id
        max_lane_id = right_entry_wp.lane_id if right_entry_wp else ego_wp.lane_id

        # Define the maximum distance and yaw difference thresholds
        # Get the maximum lane offset from the ego vehicle's lane id
        max_lane_offset = max(abs(min_lane_id - ego_wp.lane_id), abs(max_lane_id - ego_wp.lane_id))
        max_distance = self.config.leading_vehicles_max_route_distance * (1 + max_lane_offset)
        max_yaw_difference = self.config.leading_vehicles_max_route_angle_ongoing

        # Filter all leading vehicles
        leading_vehicles = self._filter_vehicles_by_route(
            ego_transform,
            ego_wp,
            planner_state,
            npc_vehicles,
            max_distance,
            max_yaw_difference,
            traffic_type="leading"
        )

        if not leading_vehicles:
            return {}

        # Generate all lanelets for each target lane waypoint
        # all_lanelets = self._get_all_lanelets(seeds)
        all_lanelets = self._get_all_lanelets(seeds, bidirectional=True)

        # Group lanelets by their corresponding lanes
        grouped_lanelets = self._group_lanelets(all_lanelets, lane_dict)

        # Group vehicles by lane name
        grouped_vehicles = self._assign_vehicles_to_lanelets(leading_vehicles, grouped_lanelets)

        return grouped_vehicles

    def get_trailing_vehicles(
        self,
        ego_transform : carla.Transform,
        ego_wp : carla.Waypoint,
        planner_state : PlannerState,
        npc_vehicles : List[carla.Vehicle],
    ) -> Dict[str, List[LaneVehicles]]:
        """
        Get the instances of vehicles trailing behind the ego vehicle.
        """
        route_waypoints = planner_state.route_waypoints[planner_state.route_index:]

        # Get the lanes in the same direction as the ego vehicle
        lane_info = LaneHandler.get_same_dir_lanes(ego_wp, route_waypoints)

        entry_corridor = lane_info["entry_corridor"]     # left?, ref, right?
        exit_corridor = lane_info["exit_corridor"]
        entry_wp = lane_info["entry_wp"]
        exit_wp = lane_info["exit_wp"]
        seeds = lane_info["seeds"]

        # Setup the source waypoints for each lanelet
        left_entry_wp = right_entry_wp = None
        left_exit_wp = right_exit_wp = None

        ego_entry_idx = entry_corridor.index(entry_wp)
        if ego_entry_idx > 0:
            left_entry_wp = entry_corridor[ego_entry_idx - 1]
        if ego_entry_idx < len(entry_corridor) - 1:
            right_entry_wp = entry_corridor[ego_entry_idx + 1]

        if exit_corridor:
            ego_exit_idx = exit_corridor.index(exit_wp)
            if ego_exit_idx > 0:
                left_exit_wp = exit_corridor[ego_exit_idx - 1]
            if ego_exit_idx < len(exit_corridor) - 1:
                right_exit_wp = exit_corridor[ego_exit_idx + 1]

        lane_dict = {
            "left":  [w for w in [left_entry_wp, left_exit_wp] if w is not None],
            "ego":   [w for w in [entry_wp, exit_wp] if w is not None],
            "right": [w for w in [right_entry_wp, right_exit_wp] if w is not None],
        }

        min_lane_id = left_entry_wp.lane_id if left_entry_wp else ego_wp.lane_id
        max_lane_id = right_entry_wp.lane_id if right_entry_wp else ego_wp.lane_id

        # Define the maximum distance and yaw difference thresholds
        # Get the maximum lane offset from the ego vehicle's lane id
        max_lane_offset = max(abs(min_lane_id - ego_wp.lane_id), abs(max_lane_id - ego_wp.lane_id))
        max_distance = self.config.trailing_vehicles_max_route_distance * (1 + max_lane_offset)
        max_yaw_difference = self.config.trailing_vehicles_max_route_angle_ongoing

        # Filter all trailing vehicles
        trailing_vehicles = self._filter_vehicles_by_route(
            ego_transform,
            ego_wp,
            planner_state,
            npc_vehicles,
            max_distance,
            max_yaw_difference,
            traffic_type="trailing"
        )

        if not trailing_vehicles:
            return {}

        # print(f'\n\nTRAILING VEHICLES:')
        # for veh in trailing_vehicles:
        #     dist = ego_transform.location.distance(veh.get_location())
        #     print(f'\tID: {veh.id}, DIST: {dist}')

        # Generate all lanelets for each target lane waypoint
        all_lanelets = self._get_all_lanelets(seeds, backward=True)

        # Group lanelets by their corresponding lanes
        grouped_lanelets = self._group_lanelets(all_lanelets, lane_dict)

        # Group vehicles by lane name
        grouped_vehicles = self._assign_vehicles_to_lanelets(trailing_vehicles, grouped_lanelets)

        return grouped_vehicles

    def get_oncoming_vehicles(
        self,
        ego_transform : carla.Transform,
        ego_wp : carla.Waypoint,
        planner_state : PlannerState,
        npc_vehicles : List[carla.Vehicle],
    ) -> Dict[str, List[LaneVehicles]]:
        """
        Get the instances of vehicles in oncoming traffic with respect to the ego vehicle
        """
        grouped_vehicles = {}
        route_index = planner_state.route_index
        route_waypoints = planner_state.route_waypoints[planner_state.route_index:]
        leading_max_detection_radius = self.config.leading_vehicles_maximum_detection_radius

        # Get the current ego waypoint
        # ego_wp = route_waypoints[route_index]

        # Get the lanes in the opposite direction as the ego vehicle
        # opp_lane_wps = LaneHandler.get_opposite_dir_lanes(ego_wp, route_waypoints)
        opp_lane_wps = LaneHandler.get_opposite_dir_lanes(route_waypoints[0], route_waypoints)

        # opp_lane_wps.append(ego_wp)
        # opp_lane_wps.append(route_waypoints[0])

        # print(f'opp_lane_wps len: {len(opp_lane_wps)}')
        # print(f'\tego_wp lane_id: {route_waypoints[0].lane_id} road_id: {route_waypoints[0].road_id}')
        # for wp in opp_lane_wps:
        #     print(f'\twp lane: {wp.lane_id} road: {wp.road_id}')

        lane_dict = {}
        if opp_lane_wps:
            # Setup the source waypoints for each lanelet
            for wp in opp_lane_wps:
                lane_name = f"oncoming-{wp.lane_id}"
                lane_dict.setdefault(lane_name, []).append(wp)

            min_lane_id = opp_lane_wps[-1].lane_id
            max_lane_id = opp_lane_wps[0].lane_id

            # Define the maximum distance and yaw difference thresholds
            # Get the maximum lane offset from the ego vehicle's lane id
            max_lane_offset = max(abs(min_lane_id - ego_wp.lane_id), abs(max_lane_id - ego_wp.lane_id))
            max_distance = self.config.trailing_vehicles_max_route_distance_lane_change
            max_yaw_difference = self.config.leading_vehicles_max_route_angle_oncoming

            # Filter all oncoming vehicles
            oncoming_vehicles = self._filter_vehicles_by_route(
                ego_transform,
                ego_wp,
                planner_state,
                npc_vehicles,
                max_distance,
                max_yaw_difference,
                traffic_type="oncoming"
            )
            # print(f'oncoming_vehicles len: {len(oncoming_vehicles)}')
            # for v in oncoming_vehicles:
            #     print(f'\tid: {v.id}')

            if not oncoming_vehicles:
                return {}

            # Generate all lanelets for each target lane waypoint
            # all_lanelets = self._get_all_lanelets(opp_lane_wps, backward=True)
            all_lanelets = self._get_all_lanelets(opp_lane_wps, bidirectional=True)

            # Group lanelets by their corresponding lanes
            grouped_lanelets = self._group_lanelets(all_lanelets, lane_dict)

            # Group vehicles by lane name
            grouped_vehicles = self._assign_vehicles_to_lanelets(oncoming_vehicles, grouped_lanelets)

        return grouped_vehicles

    def get_cross_vehicles(
        self,
        ego_transform : carla.Transform,
        ego_wp : carla.Waypoint,
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
        # ego_wp = route_waypoints[route_index]

        # Get the lanes in the perpendicular direction as the ego vehicle
        cross_lane_wps = LaneHandler.get_cross_dir_lanes(ego_wp)

        lane_dict = {}
        if cross_lane_wps:
            # Setup the source waypoints for each lanelet
            for wp in cross_lane_wps:
                lane_name = f"crossing-{wp.lane_id}"
                lane_dict.setdefault(lane_name, []).append(wp)

            # Define the maximum distance and yaw difference thresholds
            max_distance = self.config.trailing_vehicles_max_route_distance_lane_change * 10
            max_yaw_difference = self.config.leading_vehicles_max_route_angle_oncoming

            # Filter all cross vehicles
            cross_vehicles = self._filter_vehicles_by_route(
                ego_transform,
                ego_wp,
                planner_state,
                npc_vehicles,
                max_distance,
                max_yaw_difference,
                traffic_type="crossing"
            )

            if not cross_vehicles:
                return {}

            # Generate all lanelets for each target lane waypoint
            all_lanelets = self._get_all_lanelets(cross_lane_wps, backward=True)

            # Group lanelets by their corresponding lanes
            grouped_lanelets = self._group_lanelets(all_lanelets, lane_dict)

            # Group vehicles by lane name
            grouped_vehicles = self._assign_vehicles_to_lanelets(cross_vehicles, grouped_lanelets)

        return grouped_vehicles
