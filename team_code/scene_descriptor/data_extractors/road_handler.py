import numpy as np
import carla

from typing import List, Dict, Set
from dataclasses import dataclass
from collections import defaultdict

from junction_handler import JunctionHandler
from privileged_route_planner import PlannerState

@dataclass(frozen=True, slots=True)
class Lanelet:

class LaneHandler:
    """
    Identifies all ongoing and oncoming lanes
    """

    @staticmethod
    def generate_lanelet(start_wp : carla.Waypoint, max_length : float = 50.0):
        # Look ahead for any junctions
        junction_entry_wp, junction_wp = JunctionHandler.get_next_junction(start_wp, max_length, return_pre_entry=True)
        if junction_wp:
            junction_map = JunctionHandler.create_junction_map(junction_wp)
            junction_connections = JunctionHandler.get_junction_connections(junction_map, junction_entry_wp)

            for junction_lanelet in junction_connections:


    @staticmethod
    def get_same_dir_lanes(waypoint : carla.Waypoint) -> List[carla.Waypoint]:
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
    def get_opposite_dir_lanes(waypoint : carla.Waypoint) -> List[carla.Waypoint]:
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

class RoadHandler:
    """
    Identifies actors present across the road
    """

    def __init__(self, config, carla_map : carla.Map):
        self.config = config
        self.carla_map = carla_map

    def get_leading_vehicles(
            self,
            planner_state : PlannerState,
            npc_vehicles : List[carla.Vehicle],
        ) -> Dict[int, List[carla.Vehicle]]:
        """
            Get the instances of vehicles leading ahead of the ego vehicle.
        """
        route_index = planner_state.route_index
        route_waypoints = planner_state.route_waypoints
        route_points = planner_state.route_points
        rotation_angles = planner_state.rotation_angles

        leading_max_detection_radius = self.config.leading_vehicles_maximum_detection_radius

        if npc_vehicles and route_index != route_waypoints.shape[0]:
            # Get the current ego waypoint
            ego_wp = route_waypoints[route_index]

            # Get the lanes in the same direction as the ego vehicle
            same_lanes = LaneHandler.get_same_dir_lanes(ego_wp)

            # Check if the ego is near a junction
            junction_wp = None
            for i in range(min(leading_max_detection_radius, len(self.route_waypoints[self.route_index:]))):
                if self.route_waypoints[self.route_index + i].is_junction:
                    junction_wp = self.route_waypoints[self.route_index + i]
                    break

            junction_connections = self.create_junction_map(junction_wp)
            if junction_connections:
                color_entry_connection = carla.Color(255, 0, 0, 255)
                color_entry = carla.Color(255, 255, 0, 255)
                color_exit = carla.Color(0, 255, 255, 255)
                color_exit_connection = carla.Color(0, 0, 255, 255)

                same_lane_junction_wps = set()
                opposite_lane_junction_wps = set()
                # for same_lane_wp in same_lanes:
                #     print(f'Same Lane Waypoint Lane ID: {same_lane_wp.lane_id}, Road ID: {same_lane_wp.road_id}')
                #     lanelet = self.get_junction_connections(junction_connections, same_lane_wp)
                #     if lanelet:
                #         entry_wp, exit_connection_wp = lanelet
                #         same_lane_junction_wps.add(entry_wp)
                #         same_lane_junction_wps.add(exit_connection_wp)

                #         # Draw the junction waypoints
                #         self._world.debug.draw_point(same_lane_wp.transform.location + carla.Location(z=0.1), size=0.25, color=color_entry_connection, life_time=0.)
                #         # self._world.debug.draw_point(entry_wp.transform.location + carla.Location(z=0.1), size=0.25, color=color_entry, life_time=0.)
                #         # self._world.debug.draw_point(exit_connection_wp.transform.location + carla.Location(z=0.1), size=0.25, color=color_exit, life_time=0.)

                for opposite_lane_wp in opposite_lanes:
                    # print(f'Opposite Lane Waypoint Lane ID: {opposite_lane_wp.lane_id}, Road ID: {opposite_lane_wp.road_id}')
                    lanelet = self.get_junction_connections(junction_connections, opposite_lane_wp)
                    if lanelet:
                        entry_wp, exit_connection_wp = lanelet
                        opposite_lane_junction_wps.add(entry_wp)
                        opposite_lane_junction_wps.add(exit_connection_wp)

                        # Draw the junction waypoints
                        # self._world.debug.draw_point(opposite_lane_wp.transform.location + carla.Location(z=0.1), size=0.25, color=color_entry_connection, life_time=0.)
                        # self._world.debug.draw_point(entry_wp.transform.location + carla.Location(z=0.1), size=0.25, color=color_entry, life_time=0.)
                        # self._world.debug.draw_point(exit_connection_wp.transform.location + carla.Location(z=0.1), size=0.25, color=color_exit, life_time=0.)

                # junction_wps = junction_wp.get_junction().get_waypoints(carla.LaneType.Driving)
                # for entry_exit_pair in junction_wps:
                #     entry_wp = entry_exit_pair[0]
                #     exit_wp = entry_exit_pair[1]

                #     # Draw the junction waypoints
                #     self._world.debug.draw_point(entry_wp.transform.location + carla.Location(z=0.1), size=0.25, color=color_entry, life_time=0.)
                #     self._world.debug.draw_point(exit_wp.transform.location + carla.Location(z=0.1), size=0.25, color=color_exit, life_time=0.)

                same_lanes = same_lanes + list(same_lane_junction_wps)
                opposite_lanes = opposite_lanes + list(opposite_lane_junction_wps)

            # Get NPC waypoints for lane filtering
            vehicle_waypoints = [self.carla_map.get_waypoint(vehicle.get_location()) for vehicle in npc_vehicles]

            # Filter NPC vehicles based on lane direction
            valid_npc_vehicles = []
            if traffic_type == "ongoing":
                target_lanes = same_lanes
            elif traffic_type == "oncoming":
                target_lanes = opposite_lanes
                # Add the ego waypoint to the target lanes to check for oncoming vehicles invading the lane
                target_lanes.append(ego_wp)

            target_lane_road_ids = {(wp.lane_id, wp.road_id) for wp in target_lanes}
            valid_npc_vehicles = [
                (npc_vehicles[i], wp.lane_id)
                for i, wp in enumerate(vehicle_waypoints)
                if (wp.lane_id, wp.road_id) in target_lane_road_ids
            ]

            # Check if there are valid NPC vehicles
            if self.previous_leading_vehicle_ids[traffic_type].size > 0:
                # Check if previous leading vehicle IDs are still valid
                for i, vehicle in enumerate(npc_vehicles):
                    if vehicle.id in self.previous_leading_vehicle_ids[traffic_type] and (vehicle, vehicle_waypoints[i].lane_id) not in valid_npc_vehicles:
                        valid_npc_vehicles.append((vehicle, vehicle_waypoints[i].lane_id))

            if not valid_npc_vehicles:
                print(f"No valid NPC vehicles found for {traffic_type} leading traffic.")
                return {}

            # print(f"Valid NPC vehicles for {traffic_type} traffic:")
            # for vehicle, lane_id in valid_npc_vehicles:
            #     print(f"\tID: {vehicle.id}, Lane ID: {lane_id}, Road ID: {carla_map.get_waypoint(vehicle.get_location()).road_id}")
            min_lane_id = min(valid_npc_vehicles, key=lambda x: x[1])[1]
            max_lane_id = max(valid_npc_vehicles, key=lambda x: x[1])[1]

            # for vehicle, lane_id in valid_npc_vehicles:
            #     print(f"Vehicle {vehicle.id} is in lane {lane_id}")

            # Get the IDs, locations, and yaw angles of all NPC vehicles
            vehicle_ids = np.array([vehicle.id for vehicle, _ in valid_npc_vehicles])
            vehicle_locations = np.array([[vehicle.get_location().x, vehicle.get_location().y, vehicle.get_location().z] for vehicle, _ in valid_npc_vehicles])
            vehicle_yaws = np.array([vehicle.get_transform().rotation.yaw for vehicle, _ in valid_npc_vehicles])

            # Compute relative distances each NPC vehicle with the ego's route points
            # Returns a 3D array with shape (num_vehicles, num_route_points, 2)
            relative_positions = vehicle_locations[:, np.newaxis, :2] - \
            self.route_points[np.newaxis, self.route_index:self.route_index + leading_max_detection_radius, :2][:, ::self.config.points_per_meter, :]

            # Compute the relative distances
            # Returns a 2D array with shape (num_vehicles, num_route_points)
            relative_distances = np.linalg.norm(relative_positions, axis=2)

            # Get the indices of the minimum distances
            route_indices = relative_distances.argmin(axis=1)

            # Get the minimum distance for each NPC vehicle
            min_distances = relative_distances[np.arange(len(route_indices)), route_indices]

            # Get the yaw angles of the route points
            rotation_angles = self.rotation_angles[self.route_index:self.route_index + leading_max_detection_radius][::self.points_per_meter]
            route_yaws = rotation_angles[route_indices]
            yaw_differences = (route_yaws - vehicle_yaws) % 360
            yaw_differences = np.minimum(yaw_differences, 360 - yaw_differences)

            # Define the maximum distance and yaw difference thresholds
            # Get the maximum lane offset from the ego vehicle's lane id
            max_lane_offset = max(abs(min_lane_id - ego_wp.lane_id), abs(max_lane_id - ego_wp.lane_id))
            max_distance = self.config.leading_vehicles_max_route_distance * (1 + max_lane_offset)

            ego_fwd_vec = self.route_waypoints[self.route_index].transform.get_forward_vector()
            ego_fwd_vec = np.array([ego_fwd_vec.x, ego_fwd_vec.y])

            # Filter leading vehicles based on traffic type
            yaw_indices = []
            if traffic_type == "ongoing":
                max_yaw_difference = self.config.leading_vehicles_max_route_angle_ongoing
                # Compute the dot product between the ego vehicle's forward vector and the NPC vehicles' relative locations to the ego
                # Used to assert leading vehicles are in front of the ego vehicle
                ego_actor_vec = vehicle_locations[:, :2] - self.route_points[self.route_index, :2]
                loc_dot_products = np.sum(ego_actor_vec * ego_fwd_vec, axis=1)

                yaw_indices = np.where((yaw_differences < max_yaw_difference) & (loc_dot_products >= 0))[0]

            elif traffic_type == "oncoming":
                # print(f'Yaw differences: {yaw_differences}, Max yaw difference: {self.config.leading_vehicles_max_route_angle_oncoming}')
                # print(f'Distance: {min_distances}, Max distance: {max_distance}')
                max_yaw_difference = self.config.leading_vehicles_max_route_angle_oncoming

                vehicle_fwd_vecs = np.array([
                [vehicle.get_transform().get_forward_vector().x, vehicle.get_transform().get_forward_vector().y]
                for vehicle, _ in valid_npc_vehicles
                ])
                # Compute the dot product between the ego vehicle's forward vector and the NPC vehicles' forward vectors
                # Used to assert leading vehicles are traveling in the opposite direction to the ego vehicle
                heading_dot_products = np.sum(vehicle_fwd_vecs * ego_fwd_vec, axis=1)

                # Compute the dot product between the ego vehicle's forward vector and the NPC vehicles' relative locations to the ego
                # Used to assert leading vehicles are in front of the ego vehicle
                ego_actor_vec = vehicle_locations[:, :2] - self.route_points[self.route_index, :2]
                loc_dot_products = np.sum(ego_actor_vec * ego_fwd_vec, axis=1)

                # for idx, vehicle_id in enumerate(vehicle_ids):
                #    print(f'Vehicle ID: {vehicle_id}\n \tYaw Difference: {yaw_differences[idx]}, Min Distance: {min_distances[idx]}, Heading Dot Product: {heading_dot_products[idx]}, Location Dot Product: {loc_dot_products[idx]}')

                # NOTE THE YAW DIFFERENCE ONLY WORKS WHEN THE ROUTE IS STRAIGHT. DURING TURNS, THE YAW DIFFERENCE IS A LOT LOWER EVEN THOUGH THE VEHICLES CROSSING THE EGO'S PATH ARE STILL ONCOMING TRAFFIC
                # yaw_indices = np.where((yaw_differences > max_yaw_difference) & (heading_dot_products < 0) & (loc_dot_products >= 0))[0]
                yaw_indices = np.where((heading_dot_products < 0) & (loc_dot_products >= 0))[0]

            yaw_mask = np.zeros_like(vehicle_ids, dtype=bool)
            yaw_mask[yaw_indices] = True

            # Usually the road is 3.5 m wide, but in case of ParkingCrossingPedestrian it's less
            leading_vehicle_ids = vehicle_ids[(min_distances < max_distance) & yaw_mask]
            # leading_vehicle_ids = vehicle_ids[yaw_mask]
            # print(f"Leading vehicle IDs for {traffic_type} traffic: {leading_vehicle_ids}")
            self.previous_leading_vehicle_ids[traffic_type] = leading_vehicle_ids

            # Group leading vehicles by their target lane ids
            leading_vehicle_groups = {}
            for target_lane_wp in target_lanes:
                # print(f'Target Lane Waypoint Lane ID: {target_lane_wp.lane_id}, Road ID: {target_lane_wp.road_id}')
                leading_vehicle_groups[target_lane_wp.lane_id] = []
                for vehicle, lane_id in valid_npc_vehicles:
                    # print(f'\tVehicle ID: {vehicle.id}, Lane ID: {lane_id}, Road ID: {carla_map.get_waypoint(vehicle.get_location()).road_id}')
                    if vehicle.id in leading_vehicle_ids and lane_id == target_lane_wp.lane_id:
                        leading_vehicle_groups[target_lane_wp.lane_id].append(vehicle)

            return leading_vehicle_groups
        else:
            return {}

    def get_leading_vehicles(
            self,
            planner_state : PlannerState,
            npc_vehicles : List[carla.Vehicle],
        ) -> Dict[int, List[carla.Vehicle]]:
        """
            Get the instances of vehicles leading ahead of the ego vehicle.
        """
        route_index = planner_state.route_index
        route_waypoints = planner_state.route_waypoints
        route_points = planner_state.route_points
        rotation_angles = planner_state.rotation_angles

        if npc_vehicles and route_index != route_waypoints.shape[0]:
            # Get the current ego waypoint
            ego_wp = route_waypoints[route_index]
            leading_max_detection_radius = self.config.leading_vehicles_maximum_detection_radius

            # Get the lanes in the same direction and opposite direction as the ego vehicle
            same_lanes = self.get_same_dir_lanes(ego_wp)
            opposite_lanes = self.get_opposite_dir_lanes(ego_wp)

            # Check if the ego is near a junction
            junction_wp = None
            for i in range(min(leading_max_detection_radius, len(self.route_waypoints[self.route_index:]))):
                if self.route_waypoints[self.route_index + i].is_junction:
                    junction_wp = self.route_waypoints[self.route_index + i]
                    break

            junction_connections = self.create_junction_map(junction_wp)
            if junction_connections:
                color_entry_connection = carla.Color(255, 0, 0, 255)
                color_entry = carla.Color(255, 255, 0, 255)
                color_exit = carla.Color(0, 255, 255, 255)
                color_exit_connection = carla.Color(0, 0, 255, 255)

                same_lane_junction_wps = set()
                opposite_lane_junction_wps = set()
                # for same_lane_wp in same_lanes:
                #     print(f'Same Lane Waypoint Lane ID: {same_lane_wp.lane_id}, Road ID: {same_lane_wp.road_id}')
                #     lanelet = self.get_junction_connections(junction_connections, same_lane_wp)
                #     if lanelet:
                #         entry_wp, exit_connection_wp = lanelet
                #         same_lane_junction_wps.add(entry_wp)
                #         same_lane_junction_wps.add(exit_connection_wp)

                #         # Draw the junction waypoints
                #         self._world.debug.draw_point(same_lane_wp.transform.location + carla.Location(z=0.1), size=0.25, color=color_entry_connection, life_time=0.)
                #         # self._world.debug.draw_point(entry_wp.transform.location + carla.Location(z=0.1), size=0.25, color=color_entry, life_time=0.)
                #         # self._world.debug.draw_point(exit_connection_wp.transform.location + carla.Location(z=0.1), size=0.25, color=color_exit, life_time=0.)

                for opposite_lane_wp in opposite_lanes:
                    # print(f'Opposite Lane Waypoint Lane ID: {opposite_lane_wp.lane_id}, Road ID: {opposite_lane_wp.road_id}')
                    lanelet = self.get_junction_connections(junction_connections, opposite_lane_wp)
                    if lanelet:
                        entry_wp, exit_connection_wp = lanelet
                        opposite_lane_junction_wps.add(entry_wp)
                        opposite_lane_junction_wps.add(exit_connection_wp)

                        # Draw the junction waypoints
                        # self._world.debug.draw_point(opposite_lane_wp.transform.location + carla.Location(z=0.1), size=0.25, color=color_entry_connection, life_time=0.)
                        # self._world.debug.draw_point(entry_wp.transform.location + carla.Location(z=0.1), size=0.25, color=color_entry, life_time=0.)
                        # self._world.debug.draw_point(exit_connection_wp.transform.location + carla.Location(z=0.1), size=0.25, color=color_exit, life_time=0.)

                # junction_wps = junction_wp.get_junction().get_waypoints(carla.LaneType.Driving)
                # for entry_exit_pair in junction_wps:
                #     entry_wp = entry_exit_pair[0]
                #     exit_wp = entry_exit_pair[1]

                #     # Draw the junction waypoints
                #     self._world.debug.draw_point(entry_wp.transform.location + carla.Location(z=0.1), size=0.25, color=color_entry, life_time=0.)
                #     self._world.debug.draw_point(exit_wp.transform.location + carla.Location(z=0.1), size=0.25, color=color_exit, life_time=0.)

                same_lanes = same_lanes + list(same_lane_junction_wps)
                opposite_lanes = opposite_lanes + list(opposite_lane_junction_wps)

            # Get NPC waypoints for lane filtering
            vehicle_waypoints = [self.carla_map.get_waypoint(vehicle.get_location()) for vehicle in npc_vehicles]

            # Filter NPC vehicles based on lane direction
            valid_npc_vehicles = []
            if traffic_type == "ongoing":
                target_lanes = same_lanes
            elif traffic_type == "oncoming":
                target_lanes = opposite_lanes
                # Add the ego waypoint to the target lanes to check for oncoming vehicles invading the lane
                target_lanes.append(ego_wp)

            target_lane_road_ids = {(wp.lane_id, wp.road_id) for wp in target_lanes}
            valid_npc_vehicles = [
                (npc_vehicles[i], wp.lane_id)
                for i, wp in enumerate(vehicle_waypoints)
                if (wp.lane_id, wp.road_id) in target_lane_road_ids
            ]

            # Check if there are valid NPC vehicles
            if self.previous_leading_vehicle_ids[traffic_type].size > 0:
                # Check if previous leading vehicle IDs are still valid
                for i, vehicle in enumerate(npc_vehicles):
                    if vehicle.id in self.previous_leading_vehicle_ids[traffic_type] and (vehicle, vehicle_waypoints[i].lane_id) not in valid_npc_vehicles:
                        valid_npc_vehicles.append((vehicle, vehicle_waypoints[i].lane_id))

            if not valid_npc_vehicles:
                print(f"No valid NPC vehicles found for {traffic_type} leading traffic.")
                return {}

            # print(f"Valid NPC vehicles for {traffic_type} traffic:")
            # for vehicle, lane_id in valid_npc_vehicles:
            #     print(f"\tID: {vehicle.id}, Lane ID: {lane_id}, Road ID: {carla_map.get_waypoint(vehicle.get_location()).road_id}")
            min_lane_id = min(valid_npc_vehicles, key=lambda x: x[1])[1]
            max_lane_id = max(valid_npc_vehicles, key=lambda x: x[1])[1]

            # for vehicle, lane_id in valid_npc_vehicles:
            #     print(f"Vehicle {vehicle.id} is in lane {lane_id}")

            # Get the IDs, locations, and yaw angles of all NPC vehicles
            vehicle_ids = np.array([vehicle.id for vehicle, _ in valid_npc_vehicles])
            vehicle_locations = np.array([[vehicle.get_location().x, vehicle.get_location().y, vehicle.get_location().z] for vehicle, _ in valid_npc_vehicles])
            vehicle_yaws = np.array([vehicle.get_transform().rotation.yaw for vehicle, _ in valid_npc_vehicles])

            # Compute relative distances each NPC vehicle with the ego's route points
            # Returns a 3D array with shape (num_vehicles, num_route_points, 2)
            relative_positions = vehicle_locations[:, np.newaxis, :2] - \
            self.route_points[np.newaxis, self.route_index:self.route_index + leading_max_detection_radius, :2][:, ::self.config.points_per_meter, :]

            # Compute the relative distances
            # Returns a 2D array with shape (num_vehicles, num_route_points)
            relative_distances = np.linalg.norm(relative_positions, axis=2)

            # Get the indices of the minimum distances
            route_indices = relative_distances.argmin(axis=1)

            # Get the minimum distance for each NPC vehicle
            min_distances = relative_distances[np.arange(len(route_indices)), route_indices]

            # Get the yaw angles of the route points
            rotation_angles = self.rotation_angles[self.route_index:self.route_index + leading_max_detection_radius][::self.points_per_meter]
            route_yaws = rotation_angles[route_indices]
            yaw_differences = (route_yaws - vehicle_yaws) % 360
            yaw_differences = np.minimum(yaw_differences, 360 - yaw_differences)

            # Define the maximum distance and yaw difference thresholds
            # Get the maximum lane offset from the ego vehicle's lane id
            max_lane_offset = max(abs(min_lane_id - ego_wp.lane_id), abs(max_lane_id - ego_wp.lane_id))
            max_distance = self.config.leading_vehicles_max_route_distance * (1 + max_lane_offset)

            ego_fwd_vec = self.route_waypoints[self.route_index].transform.get_forward_vector()
            ego_fwd_vec = np.array([ego_fwd_vec.x, ego_fwd_vec.y])

            # Filter leading vehicles based on traffic type
            yaw_indices = []
            if traffic_type == "ongoing":
                max_yaw_difference = self.config.leading_vehicles_max_route_angle_ongoing
                # Compute the dot product between the ego vehicle's forward vector and the NPC vehicles' relative locations to the ego
                # Used to assert leading vehicles are in front of the ego vehicle
                ego_actor_vec = vehicle_locations[:, :2] - self.route_points[self.route_index, :2]
                loc_dot_products = np.sum(ego_actor_vec * ego_fwd_vec, axis=1)

                yaw_indices = np.where((yaw_differences < max_yaw_difference) & (loc_dot_products >= 0))[0]

            elif traffic_type == "oncoming":
                # print(f'Yaw differences: {yaw_differences}, Max yaw difference: {self.config.leading_vehicles_max_route_angle_oncoming}')
                # print(f'Distance: {min_distances}, Max distance: {max_distance}')
                max_yaw_difference = self.config.leading_vehicles_max_route_angle_oncoming

                vehicle_fwd_vecs = np.array([
                [vehicle.get_transform().get_forward_vector().x, vehicle.get_transform().get_forward_vector().y]
                for vehicle, _ in valid_npc_vehicles
                ])
                # Compute the dot product between the ego vehicle's forward vector and the NPC vehicles' forward vectors
                # Used to assert leading vehicles are traveling in the opposite direction to the ego vehicle
                heading_dot_products = np.sum(vehicle_fwd_vecs * ego_fwd_vec, axis=1)

                # Compute the dot product between the ego vehicle's forward vector and the NPC vehicles' relative locations to the ego
                # Used to assert leading vehicles are in front of the ego vehicle
                ego_actor_vec = vehicle_locations[:, :2] - self.route_points[self.route_index, :2]
                loc_dot_products = np.sum(ego_actor_vec * ego_fwd_vec, axis=1)

                # for idx, vehicle_id in enumerate(vehicle_ids):
                #    print(f'Vehicle ID: {vehicle_id}\n \tYaw Difference: {yaw_differences[idx]}, Min Distance: {min_distances[idx]}, Heading Dot Product: {heading_dot_products[idx]}, Location Dot Product: {loc_dot_products[idx]}')

                # NOTE THE YAW DIFFERENCE ONLY WORKS WHEN THE ROUTE IS STRAIGHT. DURING TURNS, THE YAW DIFFERENCE IS A LOT LOWER EVEN THOUGH THE VEHICLES CROSSING THE EGO'S PATH ARE STILL ONCOMING TRAFFIC
                # yaw_indices = np.where((yaw_differences > max_yaw_difference) & (heading_dot_products < 0) & (loc_dot_products >= 0))[0]
                yaw_indices = np.where((heading_dot_products < 0) & (loc_dot_products >= 0))[0]

            yaw_mask = np.zeros_like(vehicle_ids, dtype=bool)
            yaw_mask[yaw_indices] = True

            # Usually the road is 3.5 m wide, but in case of ParkingCrossingPedestrian it's less
            leading_vehicle_ids = vehicle_ids[(min_distances < max_distance) & yaw_mask]
            # leading_vehicle_ids = vehicle_ids[yaw_mask]
            # print(f"Leading vehicle IDs for {traffic_type} traffic: {leading_vehicle_ids}")
            self.previous_leading_vehicle_ids[traffic_type] = leading_vehicle_ids

            # Group leading vehicles by their target lane ids
            leading_vehicle_groups = {}
            for target_lane_wp in target_lanes:
                # print(f'Target Lane Waypoint Lane ID: {target_lane_wp.lane_id}, Road ID: {target_lane_wp.road_id}')
                leading_vehicle_groups[target_lane_wp.lane_id] = []
                for vehicle, lane_id in valid_npc_vehicles:
                    # print(f'\tVehicle ID: {vehicle.id}, Lane ID: {lane_id}, Road ID: {carla_map.get_waypoint(vehicle.get_location()).road_id}')
                    if vehicle.id in leading_vehicle_ids and lane_id == target_lane_wp.lane_id:
                        leading_vehicle_groups[target_lane_wp.lane_id].append(vehicle)

            return leading_vehicle_groups
        else:
            return {}
