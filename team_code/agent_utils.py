import numpy as np
import carla

from kinematic_bicycle_model import KinematicBicycleModel

from lateral_controller import LateralPIDController
from longitudinal_controller import LongitudinalLinearRegressionController
from privileged_route_planner import PrivilegedRoutePlanner

class AgentGeometricUtils:
    def __init__(self):
        pass

    def dot_product(self, vector1, vector2):
        """
        Calculate the dot product of two vectors.

        Args:
            vector1 (carla.Vector3D): The first vector.
            vector2 (carla.Vector3D): The second vector.

        Returns:
            float: The dot product of the two vectors.
        """
        return vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z

    def cross_product(self, vector1, vector2):
        """
        Calculate the cross product of two vectors.

        Args:
            vector1 (carla.Vector3D): The first vector.
            vector2 (carla.Vector3D): The second vector.

        Returns:
            carla.Vector3D: The cross product of the two vectors.
        """
        x = vector1.y * vector2.z - vector1.z * vector2.y
        y = vector1.z * vector2.x - vector1.x * vector2.z
        z = vector1.x * vector2.y - vector1.y * vector2.x

        return carla.Vector3D(x=x, y=y, z=z)

    def get_separating_plane(self, relative_position, plane_normal, obb1, obb2):
        """
        Check if there is a separating plane between two oriented bounding boxes (OBBs).

        Args:
            relative_position (carla.Vector3D): The relative position between the two OBBs.
            plane_normal (carla.Vector3D): The normal vector of the plane.
            obb1 (carla.BoundingBox): The first oriented bounding box.
            obb2 (carla.BoundingBox): The second oriented bounding box.

        Returns:
            bool: True if there is a separating plane, False otherwise.
        """
        # Calculate the projection of the relative position onto the plane normal
        projection_distance = abs(self.dot_product(relative_position, plane_normal))

        # Calculate the sum of the projections of the OBB extents onto the plane normal
        obb1_projection = (abs(self.dot_product(obb1.rotation.get_forward_vector() * obb1.extent.x, plane_normal)) +
                        abs(self.dot_product(obb1.rotation.get_right_vector() * obb1.extent.y, plane_normal)) +
                        abs(self.dot_product(obb1.rotation.get_up_vector() * obb1.extent.z, plane_normal)))

        obb2_projection = (abs(self.dot_product(obb2.rotation.get_forward_vector() * obb2.extent.x, plane_normal)) +
                        abs(self.dot_product(obb2.rotation.get_right_vector() * obb2.extent.y, plane_normal)) +
                        abs(self.dot_product(obb2.rotation.get_up_vector() * obb2.extent.z, plane_normal)))

        # Check if the projection distance is greater than the sum of the OBB projections
        return projection_distance > obb1_projection + obb2_projection

    def check_obb_intersection(self, obb1, obb2):
        """
        Check if two 3D oriented bounding boxes (OBBs) intersect.

        Args:
            obb1 (carla.BoundingBox): The first oriented bounding box.
            obb2 (carla.BoundingBox): The second oriented bounding box.

        Returns:
            bool: True if the two OBBs intersect, False otherwise.
        """
        relative_position = obb2.location - obb1.location

        # Check for separating planes along the axes of both OBBs
        if (self.get_separating_plane(relative_position, obb1.rotation.get_forward_vector(), obb1, obb2) or
            self.get_separating_plane(relative_position, obb1.rotation.get_right_vector(), obb1, obb2) or
            self.get_separating_plane(relative_position, obb1.rotation.get_up_vector(), obb1, obb2) or
            self.get_separating_plane(relative_position, obb2.rotation.get_forward_vector(), obb1, obb2) or
            self.get_separating_plane(relative_position, obb2.rotation.get_right_vector(), obb1, obb2) or
            self.get_separating_plane(relative_position, obb2.rotation.get_up_vector(), obb1, obb2)):

            return False

        # Check for separating planes along the cross products of the axes of both OBBs
        if (self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_forward_vector(), \
                                                            obb2.rotation.get_forward_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_forward_vector(), \
                                                            obb2.rotation.get_right_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_forward_vector(), \
                                                            obb2.rotation.get_up_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_right_vector(), \
                                                            obb2.rotation.get_forward_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_right_vector(), \
                                                            obb2.rotation.get_right_vector()), obb1, obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_right_vector(), \
                                                            obb2.rotation.get_up_vector()), obb1, obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_up_vector(), \
                                                            obb2.rotation.get_forward_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_up_vector(), \
                                                            obb2.rotation.get_right_vector()), obb1,obb2) or
            self.get_separating_plane(relative_position, self.cross_product(obb1.rotation.get_up_vector(), \
                                                            obb2.rotation.get_up_vector()), obb1, obb2)):

            return False

        # If no separating plane is found, the OBBs intersect
        return True

class AgentPrediction:
    def __init__(self, config):
        self.config = config
        self.agent_utils = AgentGeometricUtils()

        # Dummy waypoint planner
        self._waypoint_planner = PrivilegedRoutePlanner(self.config)

        # Dynamics models
        self.ego_model = KinematicBicycleModel(self.config)
        self.vehicle_model = KinematicBicycleModel(self.config)

        # Controllers
        self._turn_controller = LateralPIDController(self.config) # Lateral
        self._longitudinal_controller = LongitudinalLinearRegressionController(self.config) # Longitudinal

    def setup(self, traffic_manager, world_map, global_route_planner, ego_vehicle):
        self._traffic_manager = traffic_manager
        self._world_map = world_map
        self._global_route_planner = global_route_planner
        self._ego_vehicle = ego_vehicle

    def update_state(self, vehicle_context, ego_context):
        self.vehicle_context = vehicle_context
        self.ego_context = ego_context

        self.npc_vehicle_dict = {
            vehicle.id: vehicle for vehicle in vehicle_context['npc_vehicles']
        }

    def _group_npc_vehicles_by_road_and_lane(self, vehicles_list):
        """
        Group NPC vehicles by road and lane.

        Args:
            vehicles_list (list): List of NPC vehicles.

        Returns:
            dict: A dictionary with road IDs as keys and dictionaries as values,
                  where each dictionary has lane IDs as keys and lists of NPC vehicles in each lane as values.
        """
        vehicles_by_road_and_lane = {}
        for vehicle in vehicles_list:
            # Locate the NPC vehicle in the map
            vehicle_loc = vehicle.get_location()
            vehicle_wp = self._world_map.get_waypoint(vehicle_loc)

            if not vehicle_wp:
                print(f'Could not find waypoint for vehicle {vehicle.id}')
                continue

            # Get the road and lane ID of the NPC vehicle
            road_id = vehicle_wp.road_id
            lane_id = vehicle_wp.lane_id

            if road_id not in vehicles_by_road_and_lane:
                vehicles_by_road_and_lane[road_id] = {}
            if lane_id not in vehicles_by_road_and_lane[road_id]:
                vehicles_by_road_and_lane[road_id][lane_id] = []

            vehicles_by_road_and_lane[road_id][lane_id].append(vehicle)

        return vehicles_by_road_and_lane

    def _find_waypoint_entry_exit(self, waypoint):
        """
        Find the entry and exit points of the waypoint's corresponding road segment.
        Args:
            waypoint (carla.Waypoint): The waypoint.

        Returns:
            tuple: A tuple containing the entry and exit points of the road segment.
        """
        entry_xyz, exit_xyz = None, None
        try:
            entry_xyz, exit_xyz = self._global_route_planner._road_id_to_edge[waypoint.road_id][waypoint.section_id][waypoint.lane_id]
        except KeyError:
            print(f'Could not find edge for waypoint {waypoint.transform.location}')
            pass
        return (entry_xyz, exit_xyz)

    def _is_connected_edge(self, entry_exit_pair, next_entry_exit_pair):
        """
        Check if two edges are connected.

        Args:
            entry_exit_pair (tuple): The entry and exit points of the first edge.
            next_entry_exit_pair (tuple): The entry and exit points of the second edge.

        Returns:
            bool: True if the edges are connected, False otherwise.
        """
        edge = self._global_route_planner._graph.edges[entry_exit_pair[0], entry_exit_pair[1]]
        next_edge = self._global_route_planner._graph.edges[next_entry_exit_pair[0], next_entry_exit_pair[1]]

        exit_wp = edge['exit_waypoint']
        next_entry_wp = next_edge['entry_waypoint']

        return exit_wp.road_id == next_entry_wp.road_id and \
               exit_wp.section_id == next_entry_wp.section_id

    def _group_waypoints_by_road_ids(self, waypoints):
        """
        Prune a list of waypoints to only contain unique road IDs.

        Args:
            waypoints (list): List of waypoints.

        Returns:
            pruned_waypoints (list): List of waypoints containing unique road IDs.
        """
        if not waypoints:
            return []

        pruned_waypoints = [waypoints[0]]
        for waypoint in waypoints:
            if waypoint.road_id != pruned_waypoints[-1].road_id:
                pruned_waypoints.append(waypoint)

        return pruned_waypoints

    def _get_next_ego_waypoints(self, ego_route, ego_speed):
        """
        Get the next waypoints of the ego vehicle.

        Args:
            ego_route (list): List of waypoints of the ego vehicle.
            ego_speed (float): Speed of the ego vehicle.

        Returns:
            list: List of waypoints of the ego vehicle up to the specified distance.
        """
        if ego_route:
            distance = ego_speed * self.config.prediction_horizon
            traveled_distance = 0.0
            cur_index = 0
            cur_position = ego_route[cur_index].transform.location

            while traveled_distance < distance and cur_index + 1 < len(ego_route):
                traveled_distance = cur_position.distance(ego_route[cur_index + 1].transform.location)
                cur_index += 1

            return ego_route[:cur_index]
        return None

    def _get_ego_route_entry_exit_pairs(self, ego_waypoints):
        """
        Get the entry and exit points of the road segments of the ego vehicle's route.

        Args:
            ego_waypoints (list): List of waypoints of the ego vehicle.

        Returns:
            list: List of tuples containing the entry and exit points of the road segments
        """
        ego_entry_exit_pairs = []
        pruned_ego_waypoints = self._group_waypoints_by_road_ids(ego_waypoints)
        for wp in pruned_ego_waypoints:
            ego_entry_exit_pairs.append(self._find_waypoint_entry_exit(wp))

        return ego_entry_exit_pairs

    def _choose_at_junction(self, waypoint_choices, ego_entry_exit_pairs):
        """
        Choose the next waypoint at a junction.

        Args:
            waypoint_choices (list): List of waypoint choices at a junction.
            ego_entry_exit_pairs (list): List of entry and exit points of the ego vehicle's route.

        Returns:
            carla.Waypoint: The chosen waypoint.
        """
        for wp in waypoint_choices:
            entry_exit_pair = self._find_waypoint_entry_exit(wp)
            for ego_entry_exit_pair in ego_entry_exit_pairs:
                if self._is_connected_edge(entry_exit_pair, ego_entry_exit_pair):
                    return wp

        return waypoint_choices[np.random.randint(len(waypoint_choices))] # Randomly choose a waypoint

    def _get_distance(self, wp1, wp2):
        return wp1.transform.location.distance(wp2.transform.location)

    def _interpolate_between_waypoints(self, source_wp, target_wp):
        """
        Interpolate waypoints between source and target waypoints up to a desired distance.

        Args:
            source_wp (carla.Waypoint): Source waypoint.
            target_wp (carla.Waypoint): Target waypoint.

        Returns:
            list: List of interpolated waypoints between source and target waypoints up to desired distance.
            float: The total distance traveled.
        """
        # print(f'Interpolating between waypoints {source_wp.transform.location} and {target_wp.transform.location}')
        interp_wps = []
        traveled_distance = 0.0
        cur_wp = source_wp

        while self._get_distance(cur_wp, target_wp) > self.config.sampling_resolution and self._get_distance(cur_wp, source_wp) < self._get_distance(target_wp, source_wp):
            # print(f'traveled_distance: {traveled_distance}')
            wp_choice = cur_wp.next(self.config.sampling_resolution)
            if len(wp_choice) > 1:
                max_dot = -1 * np.inf
                target_wp_vec = target_wp.transform.get_forward_vector()
                for wp in wp_choice:
                    wp_vec = wp.transform.get_forward_vector()
                    target_select_wp_dot = target_wp_vec.dot(wp_vec)
                    # Select waypoint with straightest path to target waypoint
                    if target_select_wp_dot > max_dot:
                        max_dot = target_select_wp_dot
                        cur_wp = wp
            else:
                cur_wp = wp_choice[0]

            interp_wps.append(cur_wp)
            traveled_distance = self._get_distance(cur_wp, source_wp)

        interp_wps.append(target_wp)

        return interp_wps, traveled_distance

    def _get_waypoint_list_at_distance(self, waypoint, distance, ego_entry_exit_pairs):
        """
        Get a list of waypoints at a certain distance from the input waypoint.

        Args:
            waypoint (carla.Waypoint): The input waypoint.
            distance (float): The distance to travel.
            ego_entry_exit_pairs (list): List of entry and exit points of the ego vehicle's route.

        Returns:
            list: List of waypoints at the specified distance.
        """
        # print(f'Getting waypoints at distance {distance} from {waypoint.transform.location}')
        traveled_distance = 0.0
        plan = []

        cur_wp = waypoint
        while traveled_distance < distance:
            # print(f'traveled_distance: {traveled_distance}')
            wp_choice = cur_wp.next(self.config.sampling_resolution)
            if len(wp_choice) > 1:
                # print(f'Junction detected at waypoint {cur_wp.transform.location}')
                cur_wp = self._choose_at_junction(wp_choice, ego_entry_exit_pairs)
            else:
                cur_wp = wp_choice[0]

            plan.append(cur_wp)
            traveled_distance = self._get_distance(cur_wp, waypoint)

        return plan

    def predict_npc_vehicle_waypoints(self, ego_data, nearby_npc_vehicles_list):
        """
        Get the predicted waypoints of NPC vehicles with a 3s prediction horizon.

        Args:
            ego_data (dict): Dictionary containing the ego vehicle's data.
            nearby_npc_vehicles_list (list): List of nearby NPC vehicles.

        Returns:
            dict: A dictionary containing the predicted positions of NPC vehicles.
        """
        ego_route = ego_data['route']
        ego_speed = ego_data['speed']

        # Get the next waypoints of the ego vehicle
        ego_waypoints = self._get_next_ego_waypoints(ego_route, ego_speed)

        # Get the entry and exit points of the road segments of the ego vehicle's route
        ego_entry_exit_pairs = self._get_ego_route_entry_exit_pairs(ego_waypoints)

        predicted_positions = {}
        npc_vehicle_dict = {}
        for vehicle in nearby_npc_vehicles_list:
            vehicle_plan = []

            # Get vehicle data
            vehicle_loc = vehicle.get_location()
            vehicle_speed = vehicle.get_velocity().length()
            vehicle_predicted_distance = vehicle_speed * self.config.prediction_horizon

            # Locate NPC vehicle waypoint in the map
            vehicle_wp = self._world_map.get_waypoint(vehicle_loc)

            # Try to get the next actions of the NPC vehicle from TrafficManager
            next_vehicle_actions = None
            try:
                next_vehicle_actions = self._traffic_manager.get_all_actions(vehicle)
            except Exception as e:
                print(f'Error getting next actions for vehicle {vehicle.id}: {e}')
                print(f'Using current waypoint instead')

            # Vehicle not controlled by TrafficManager (either static vehicle or scenario vehicle)
            if not next_vehicle_actions:
                vehicle_plan = [vehicle_wp]
                # Check if scenario vehicle
                # TODO: Implement scenario vehicle check
                pass

            elif len(next_vehicle_actions) > 1:
                next_vehicle_wps = [action[1] for action in next_vehicle_actions]
                source_wp = vehicle_wp

                for next_target_wp in next_vehicle_wps:
                    next_target_wp_dist = self._get_distance(next_target_wp, source_wp)
                    if next_target_wp_dist > self.config.sampling_resolution:
                        # Interpolate waypoints between current and next waypoint
                        interp_plan, traveled_dist = self._interpolate_between_waypoints(source_wp, next_target_wp)
                        vehicle_plan.extend(interp_plan)
                        vehicle_predicted_distance -= traveled_dist
                    else:
                        vehicle_plan.append(next_target_wp)
                        vehicle_predicted_distance -= next_target_wp_dist

                    source_wp = next_target_wp

                # print(f'Preliminary plan for vehicle {vehicle.id}:')
                # for wp in vehicle_plan:
                #     print(f'    Location: {wp.transform.location}, Road ID: {wp.road_id}, Lane ID: {wp.lane_id}')
                vehicle_wp = vehicle_plan[-1]

            else:
                next_target_wp = next_vehicle_actions[0][1]
                next_target_wp_dist = self._get_distance(next_target_wp, vehicle_wp)
                if next_target_wp_dist > self.config.sampling_resolution:
                    # Interpolate waypoints between current and next waypoint
                    interp_plan, traveled_dist = self._interpolate_between_waypoints(vehicle_wp, next_target_wp)
                    vehicle_plan.extend(interp_plan)
                    vehicle_predicted_distance -= traveled_dist
                else:
                    vehicle_plan.append(next_target_wp)
                    vehicle_predicted_distance -= next_target_wp_dist

                vehicle_wp = vehicle_plan[-1]

            # Get waypoints at distance from the vehicle's current waypoint
            if vehicle_predicted_distance > 0:
                vehicle_plan.extend(self._get_waypoint_list_at_distance(vehicle_wp, vehicle_predicted_distance, ego_entry_exit_pairs))

            # print(f'Predicted path for vehicle {vehicle.id}:')
            # for wp in vehicle_plan:
            #     print(f'    Location: {wp.transform.location}, Road ID: {wp.road_id}, Lane ID: {wp.lane_id}')

            if vehicle.id not in predicted_positions:
                predicted_positions[vehicle.id] = []
            predicted_positions[vehicle.id].extend(vehicle_plan)
            npc_vehicle_dict[vehicle.id] = vehicle

        return predicted_positions, npc_vehicle_dict

    def _get_nearest_vehicle_route_point(self, vehicle_route_points, vehicle_location, route_index):
        """
        Get the nearest route point to the vehicle.

        Args:
            vehicle_route_points (numpy.ndarray): An array of waypoints representing the planned route.
            vehicle_location (numpy.ndarray): The current location of the vehicle.

        Returns:
            numpy.ndarray: The remaining route points from the nearest route point to the vehicle.
        """
        to_index = self.config.ego_vehicles_route_point_search_distance
        search_range = min(route_index + to_index, vehicle_route_points.shape[0])

        # Find the index of the nearest route point to the agent's position
        route_index_offset = np.argmin(np.linalg.norm(
            vehicle_location[None, :2] - vehicle_route_points[route_index:search_range, :2], axis=1)
        )

        return vehicle_route_points[route_index + route_index_offset:], route_index_offset

    def forecast_ego_vehicle_bounding_boxes(self, ego_data, target_speed, num_future_frames):
        """
            Forecast the future states of the ego agent using the kinematic bicycle model and assume their is no hazard to
            check subsequently whether the ego vehicle would collide.

            Args:
                current_ego_transform (carla.Transform): The current transform of the ego vehicle.
                current_ego_speed (float): The current speed of the ego vehicle in m/s.
                num_future_frames (int): The number of future frames to forecast.
                initial_target_speed (float): The initial target speed for the ego vehicle.
                route_points (numpy.ndarray): An array of waypoints representing the planned route.

            Returns:
                list: A list of bounding boxes representing the future states of the ego vehicle.
            """
        self._turn_controller.save_state()

        # Initialize the initial state without braking
        ego_location = np.array(
            [ego_data['location'].x, ego_data['location'].y, ego_data['location'].z])
        ego_heading_angle = np.array([ego_data['compass']])
        ego_speed = np.array([ego_data['speed']])

        ego_target_speed = target_speed
        ego_route_points = ego_data['route_points']

        # Calculate the throttle command based on the target speed and current speed
        throttle = self._longitudinal_controller.get_throttle_extrapolation(ego_target_speed, ego_speed)
        steering = self._turn_controller.step(ego_route_points, ego_speed, ego_location, ego_heading_angle.item())
        action = np.array([steering, throttle, 0.0]).flatten()

        future_bounding_boxes = []
        # Iterate over the future frames and forecast the ego agent's state
        route_index = 0
        for i in range(num_future_frames):
            # Forecast the next state using the kinematic bicycle model
            ego_location, ego_heading_angle, ego_speed = self.ego_model.forecast_ego_vehicle(ego_location, ego_heading_angle, ego_speed, action)

            # Update the route and extrapolate steering and throttle commands
            ego_forecast_route, route_index_offset = self._get_nearest_vehicle_route_point(ego_route_points, ego_location, route_index)
            route_index += route_index_offset

            steering = self._turn_controller.step(ego_forecast_route, ego_speed, ego_location, ego_heading_angle.item())

            throttle = self._longitudinal_controller.get_throttle_extrapolation(ego_target_speed, ego_speed)
            action = np.array([steering, throttle, 0.0]).flatten()

            # Calculate the heading angle in degrees
            ego_heading_angle_degrees = np.rad2deg(ego_heading_angle).item()

            # Decrease the ego vehicles bounding box if it is slow and resolve permanent bounding box
            # intersectinos at collisions.
            # In case of driving increase them for safety.
            extent = self._ego_vehicle.bounding_box.extent
            # Otherwise we would increase the extent of the bounding box of the vehicle
            extent = carla.Vector3D(x=extent.x, y=extent.y, z=extent.z)
            extent.x *= self.config.slow_speed_extent_factor_ego \
                if ego_speed < self.config.extent_ego_bbs_speed_threshold \
                else self.config.high_speed_extent_factor_ego_x
            extent.y *= self.config.slow_speed_extent_factor_ego \
                if ego_speed < self.config.extent_ego_bbs_speed_threshold \
                else self.config.high_speed_extent_factor_ego_y

            ego_carla_location = carla.Location(x=ego_location[0].item(), y=ego_location[1].item(), z=ego_location[2].item())
            ego_bounding_box = carla.BoundingBox(ego_carla_location, extent)
            ego_bounding_box.rotation = carla.Rotation(pitch=0, yaw=ego_heading_angle_degrees, roll=0)

            future_bounding_boxes.append(ego_bounding_box)

        self._turn_controller.load_state()

        return future_bounding_boxes

    def forecast_npc_vehicle_bounding_boxes(self, npc_vehicles_dict, npc_predicted_paths, num_future_frames):
        """
            Forecast the future states of the NPC agents using the kinematic bicycle model.

            Args:
                npc_predicted_paths (dict): A dictionary containing the predicted waypoints of NPC vehicles.
                num_future_frames (int): The number of future frames to forecast.

            Returns:
                dict: A dictionary containing the future states of the NPC agents.
            """

        npc_vehicles_future_bounding_boxes_dict = {}
        for vehicle_id, predicted_path in npc_predicted_paths.items():
            if len(predicted_path) > 1:
                self._turn_controller.save_state()

                vehicle = npc_vehicles_dict[vehicle_id]
                vehicle_location = np.array(
                    [vehicle.get_location().x, vehicle.get_location().y, vehicle.get_location().z]
                )
                vehicle_heading_angle = np.array([np.deg2rad(vehicle.get_transform().rotation.yaw)])
                vehicle_speed = np.array([vehicle.get_velocity().length()])

                vehicle_target_speed = vehicle_speed

                vehicle_route_points = [wp.transform.location for wp in predicted_path]
                vehicle_route_points = np.array([[loc.x, loc.y, loc.z] for loc in vehicle_route_points])
                vehicle_route_points, _ = self._waypoint_planner.smooth_and_supersample(vehicle_route_points)

                # Calculate the throttle command based on the target speed and current speed
                throttle = self._longitudinal_controller.get_throttle_extrapolation(vehicle_target_speed, vehicle_speed)
                steering = self._turn_controller.step(vehicle_route_points, vehicle_speed, vehicle_location, vehicle_heading_angle.item())
                action = np.array([steering, throttle, 0.0]).flatten()

                future_bounding_boxes = []
                # Iterate over the future frames and forecast the npc vehicle's state
                route_index = 0
                for i in range(num_future_frames):
                    # Forecast the next state using the kinematic bicycle model
                    vehicle_location, vehicle_heading_angle, vehicle_speed = self.vehicle_model.forecast_ego_vehicle(vehicle_location, vehicle_heading_angle, vehicle_speed, action)

                    # Update the route and extrapolate steering and throttle commands
                    vehicle_forecast_route, route_index_offset = self._get_nearest_vehicle_route_point(vehicle_route_points, vehicle_location, route_index)
                    route_index += route_index_offset

                    steering = self._turn_controller.step(vehicle_forecast_route, vehicle_speed, vehicle_location, vehicle_heading_angle.item())
                    throttle = self._longitudinal_controller.get_throttle_extrapolation(vehicle_target_speed, vehicle_speed)
                    action = np.array([steering, throttle, 0.0]).flatten()

                    # Calculate the heading angle in degrees
                    vehicle_heading_angle_degrees = np.rad2deg(vehicle_heading_angle).item()

                    # Decrease the NPC vehicles bounding box if it is slow and resolve permanent bounding box
                    # intersectinos at collisions.
                    # In case of driving increase them for safety.
                    extent = vehicle.bounding_box.extent
                    # Otherwise we would increase the extent of the bounding box of the vehicle
                    extent = carla.Vector3D(x=extent.x, y=extent.y, z=extent.z)
                    extent.x *= self.config.slow_speed_extent_factor_ego \
                        if vehicle_speed < self.config.extent_ego_bbs_speed_threshold \
                        else self.config.high_speed_extent_factor_ego_x
                    extent.y *= self.config.slow_speed_extent_factor_ego \
                        if vehicle_speed < self.config.extent_ego_bbs_speed_threshold \
                        else self.config.high_speed_extent_factor_ego_y

                    vehicle_carla_location = carla.Location(x=vehicle_location[0].item(), y=vehicle_location[1].item(), z=vehicle_location[2].item())
                    vehicle_bounding_box = carla.BoundingBox(vehicle_carla_location, extent)
                    vehicle_bounding_box.rotation = carla.Rotation(pitch=0, yaw=vehicle_heading_angle_degrees, roll=0)

                    future_bounding_boxes.append(vehicle_bounding_box)

                self._turn_controller.load_state()
                npc_vehicles_future_bounding_boxes_dict[vehicle_id] = future_bounding_boxes

        return npc_vehicles_future_bounding_boxes_dict

    def check_ego_collision_vehicles(self, near_lane_change, ego_bounding_boxes, npc_vehicles_future_bounding_boxes_dict, npc_vehicles_lanes):
        """
        Check if the ego vehicle will collide with any NPC vehicles in the future.

        Args:
            near_lane_change (bool): Flag indicating if the ego vehicle is near a lane change.
            ego_bounding_boxes (list): A list of bounding boxes representing the future states of the ego vehicle.
            npc_vehicles_future_bounding_boxes_dict (dict): A dictionary containing the future states of the NPC agents.
            npc_vehicles_lanes (dict): A dictionary containing the lanes of the NPC vehicles.
        Returns:

        """
        npc_vehicle_collisions = {}

        ego_vehicle_location = self.ego_context['location']
        ego_speed = self.ego_context['speed']

        leading_vehicle_ids = npc_vehicles_lanes['ego']['leading_vehicles']
        trailing_vehicle_ids = npc_vehicles_lanes['ego']['trailing_vehicles']

        for i, ego_bounding_box in enumerate(ego_bounding_boxes):
            for vehicle_id, future_bounding_boxes in npc_vehicles_future_bounding_boxes_dict.items():
                # Skip leading and rear vehicles if not near a lane change
                if vehicle_id in leading_vehicle_ids and not near_lane_change:
                    continue
                elif vehicle_id in trailing_vehicle_ids and not near_lane_change:
                    continue
                else:
                    # Check if the ego bounding box intersects with the predicted bounding box of the actor
                    intersects_with_ego = self.agent_utils.check_obb_intersection(ego_bounding_box, future_bounding_boxes[i])

                    if intersects_with_ego:
                        target_vehicle = self.npc_vehicle_dict[vehicle_id]
                        collision_point = future_bounding_boxes[i].location

                        vehicle_spacing = max(2 * target_vehicle.bounding_box.extent.x, 2 * ego_bounding_box.extent.x)
                        distance_to_collision = collision_point.distance(ego_vehicle_location) - vehicle_spacing
                        time_to_collision = distance_to_collision / ego_speed

                        collision_entry = {
                            "vehicle_id": vehicle_id,
                            "collision_point": collision_point,
                            "distance_to_collision": distance_to_collision,
                            "time_to_collision": time_to_collision,
                        }
                        if vehicle_id not in npc_vehicle_collisions:
                            npc_vehicle_collisions[vehicle_id] = collision_entry

        return npc_vehicle_collisions

    def check_ego_collisions_cyclists(self, ego_bounding_boxes, npc_cyclists_future_bounding_boxes_dict):
        pass

    def check_ego_collisions_peds(self, ego_bounding_boxes, npc_peds_future_bounding_boxes_dict):
        pass
