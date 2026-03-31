import carla
import numpy as np

from typing import List, Tuple, Dict, Optional

from config import GlobalConfig
from scene_descriptor.data_extractors.vehicle_data_extractor import VehicleDataEntry
from kinematic_bicycle_model import KinematicBicycleModel
from lateral_controller import LateralPIDController
from longitudinal_controller import LongitudinalLinearRegressionController

class MotionForecaster:
    def __init__(self, config : GlobalConfig):
        self.config = config
        self.steps_per_output = self.config.bicycle_frame_rate // self.config.prediction_frequency
        self.vehicle_model = KinematicBicycleModel(self.config)

        self.lateral_controller = LateralPIDController(self.config)
        self.long_controller = LongitudinalLinearRegressionController(self.config)

    def _get_all_nearest_route_indices(
        self,
        locations : np.ndarray,
        route_list : List[np.ndarray],
        search_dist : int
    ) -> np.ndarray:
        """
        Vectorized search for the nearest point for ALL vehicles.
        """
        num_actors = locations.shape[0]

        # 1. Create a padded array of the next 'search_dist' points for each vehicle
        # Shape: (num_actors, search_dist, 2)
        search_block = np.zeros((num_actors, search_dist, 2))

        for idx, route in enumerate(route_list):
            # Slice the next N points; if route is shorter, pad with the last point
            available_pts = route[:search_dist, :2]
            n_pts = available_pts.shape[0]
            if n_pts == 0:
                search_block[idx, :, :] = locations[idx, :2]  # or zeros / last known point
                continue

            search_block[idx, :n_pts] = available_pts
            if n_pts < search_dist:
                search_block[idx, n_pts:] = available_pts[-1] # Pad with last point

        # 2. Compute distances: (num_actors, 1, 2) vs (num_actors, search_dist, 2)
        # Resulting diffs shape: (num_actors, search_dist, 2)
        diffs = search_block - locations[:, np.newaxis, :2]

        # 3. Squared Euclidean distance (faster than sqrt)
        dists_sq = np.sum(diffs**2, axis=2) # Shape: (num_actors, search_dist)

        # 4. Find the index of the minimum distance for each actor
        return np.argmin(dists_sq, axis=1)

    def _get_nearest_vehicle_route_point(
        self,
        vehicle_route_points : np.ndarray,
        vehicle_location : np.ndarray,
        route_index : int
    ) -> Tuple[np.ndarray, int]:
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

    def _get_braking_distance(
        self,
        speed: float,
        reaction_time: float,
        brake_accel: float,
    ) -> float:
        d_brake = speed * reaction_time + (speed ** 2) / (2 * brake_accel)
        return d_brake

    def _dilate_bbox(
        self,
        bbox: carla.BoundingBox,
        front_buffer_m: float,
        rear_buffer_m: float,
        lateral_buffer_m: float,
    ) -> carla.BoundingBox:
        """
        Dilate a CARLA bounding box asymmetrically along its longitudinal axis,
        and symmetrically along its lateral axis.

        Args:
            bbox:
                Input bounding box. Assumed to already have the correct world-space
                location and rotation.
            front_buffer_m:
                Extra distance to add in front of the vehicle (meters).
            rear_buffer_m:
                Extra distance to add behind the vehicle (meters).
            lateral_buffer_m:
                Symmetric expansion on the left and right sides of the vehicle (meters).

        Returns:
            A new carla.BoundingBox with updated center and extents.
        """

        # Original half-extents
        ex = bbox.extent.x
        ey = bbox.extent.y
        ez = bbox.extent.z

        # New half-extents
        # X is asymmetrical, so the half-extent grows by the average of the two buffers
        new_ex = ex + 0.5 * (front_buffer_m + rear_buffer_m)

        # Y is symmetrical, so the half-extent simply grows by the lateral buffer amount
        new_ey = ey + lateral_buffer_m

        # Center shift in the box's LOCAL forward direction (+x in CARLA bbox space)
        local_forward_shift = 0.5 * (front_buffer_m - rear_buffer_m)

        # Rotate that local shift into world XY using the bbox yaw
        yaw_rad = np.deg2rad(bbox.rotation.yaw)
        dx_world = local_forward_shift * np.cos(yaw_rad)
        dy_world = local_forward_shift * np.sin(yaw_rad)

        new_loc = carla.Location(
            x=bbox.location.x + dx_world,
            y=bbox.location.y + dy_world,
            z=bbox.location.z,
        )

        new_extent = carla.Vector3D(
            x=new_ex,
            y=new_ey,
            z=ez,
        )

        dilated_bbox = carla.BoundingBox(new_loc, new_extent)
        dilated_bbox.rotation = bbox.rotation

        return dilated_bbox

    # def forecast_vehicle_bbs_array(
    #     self,
    #     all_vehicle_data : List[VehicleDataEntry],
    #     prediction_horizons : Dict[int, int],
    #     default_future_frames : int
    # ) -> Dict[int, List[carla.BoundingBox]]:
    #     predicted_bounding_boxes = {v.id : [] for v in all_vehicle_data}

    #     # Setup initial vectorized states (control, velocity, location, heading)
    #     previous_actions = np.array([[v.steer, v.throttle, v.brake] for v in all_vehicle_data])
    #     velocities = np.array([v.speed for v in all_vehicle_data])
    #     locations = np.array([[v.x, v.y, v.z] for v in all_vehicle_data])
    #     headings = np.array([v.heading for v in all_vehicle_data])

    #     # Setup prediction horizons for masking
    #     active_horizons = np.array([prediction_horizons.get(v.id, default_future_frames) for v in all_vehicle_data])

    #     # Cache vehicle extents
    #     extents = [v.vehicle.bounding_box.extent for v in all_vehicle_data]

    #     # Forecast future locations, headings, velocities and create their bounding boxes
    #     for i in range(default_future_frames):
    #         # KBM forecast
    #         if i != 0:
    #             next_locs, next_heads, next_vels = self.vehicle_model.forecast_other_vehicles(
    #                 locations, headings, velocities, previous_actions
    #             )

    #             # Update vehicles that still have an active prediction horizon
    #             mask = i < active_horizons

    #             # Update states (if mask is False, keep previous state)
    #             locations[mask] = next_locs[mask]
    #             headings[mask] = next_heads[mask]
    #             velocities[mask] = next_vels[mask]

    #         # Vehicle predicted bounding boxes construction
    #         if i % self.steps_per_output == 0:
    #             yaws_deg = np.rad2deg(headings)

    #             for idx, v_data in enumerate(all_vehicle_data):
    #                 loc = carla.Location(
    #                     x=float(locations[idx, 0]),
    #                     y=float(locations[idx, 1]),
    #                     z=float(locations[idx, 2]),
    #                 )
    #                 rot = carla.Rotation(pitch=0, yaw=float(yaws_deg[idx]), roll=0)

    #                 # Create bounding box
    #                 e = extents[idx]
    #                 bbox = carla.BoundingBox(loc, e)
    #                 bbox.rotation = rot

    #                 predicted_bounding_boxes[v_data.id].append(bbox)

    #     return predicted_bounding_boxes

    # def forecast_vehicle_bbs_array(
    #     self,
    #     all_vehicle_data : List[VehicleDataEntry],
    #     prediction_horizons : Dict[int, int],
    #     default_future_frames : int
    # ) -> Dict[int, List[carla.BoundingBox]]:
    #     self.lateral_controller.reset_state()

    #     predicted_bounding_boxes = {v.id : [] for v in all_vehicle_data}

    #     # Setup initial vectorized states (control, velocity, location, heading)
    #     previous_actions = np.array([[v.steer, v.throttle, v.brake] for v in all_vehicle_data])
    #     velocities = np.array([v.speed for v in all_vehicle_data])
    #     locations = np.array([[v.x, v.y, v.z] for v in all_vehicle_data])
    #     headings = np.array([v.heading for v in all_vehicle_data])

    #     # Setup prediction horizons for masking
    #     active_horizons = np.array([prediction_horizons.get(v.id, default_future_frames) for v in all_vehicle_data])

    #     # Cache vehicle extents
    #     extents = [v.vehicle.bounding_box.extent for v in all_vehicle_data]

    #     # Get route points per vehicle
    #     route_pts_by_idx : List[np.ndarray] = [v.lanelet.dense_points[v.lanelet_route_idx:] for v in all_vehicle_data]

    #     # PID controller history per vehicle
    #     pid_hist_by_vid : Dict[int, List[float]] = {v.id : [] for v in all_vehicle_data}

    #     def should_use_controller(
    #         route_pts_xy : np.ndarray,
    #         vehicle_pos_xy : np.ndarray,
    #     ) -> bool:
    #         if route_pts_xy.shape[0] < 2:
    #             return False

    #         end_pt = route_pts_xy[-1]
    #         prev_end_pt = route_pts_xy[-2]
    #         last_dir = end_pt - prev_end_pt
    #         n = np.hypot(last_dir[0], last_dir[1])
    #         last_dir_unit = last_dir / n

    #         beyond_end = np.dot(vehicle_pos_xy - end_pt, last_dir_unit) > 0.0

    #         return not beyond_end

    #     # Forecast future locations, headings, velocities and create their bounding boxes
    #     for i in range(default_future_frames):
    #         if i != 0:
    #             nearest_offsets = self._get_all_nearest_route_indices(locations, route_pts_by_idx, self.config.ego_vehicles_route_point_search_distance)

    #             # Update steer using lateral controller where valid, otherwise use previous command
    #             for idx, v_data in enumerate(all_vehicle_data):
    #                 if i >= active_horizons[idx]:
    #                     continue

    #                 # Slice the route based on the vectorized search result
    #                 offset = nearest_offsets[idx]
    #                 route_pts_by_idx[idx] = route_pts_by_idx[idx][offset:]
    #                 current_route = route_pts_by_idx[idx]

    #                 if should_use_controller(current_route[:, :2], locations[idx, :2]):
    #                     # Context Switch PID
    #                     self.lateral_controller.error_history = pid_hist_by_vid[v_data.id]

    #                     steer = self.lateral_controller.step(
    #                         route_points=current_route,
    #                         current_speed=velocities[idx],
    #                         vehicle_position=locations[idx],
    #                         vehicle_heading=headings[idx]
    #                     )

    #                     pid_hist_by_vid[v_data.id] = self.lateral_controller.error_history
    #                     previous_actions[idx, 0] = steer

    #             # Vectorized Physics Update
    #             next_locs, next_heads, next_vels = self.vehicle_model.forecast_other_vehicles(
    #                 locations, headings, velocities, previous_actions
    #             )

    #             # Update vehicles that still have an active prediction horizon
    #             mask = i < active_horizons

    #             # Update states (if mask is False, keep previous state)
    #             locations[mask] = next_locs[mask]
    #             headings[mask] = next_heads[mask]
    #             velocities[mask] = next_vels[mask]

    #         # Vehicle predicted bounding boxes construction
    #         if i % self.steps_per_output == 0:
    #             yaws_deg = np.rad2deg(headings)

    #             for idx, v_data in enumerate(all_vehicle_data):
    #                 loc = carla.Location(
    #                     x=float(locations[idx, 0]),
    #                     y=float(locations[idx, 1]),
    #                     z=float(locations[idx, 2]),
    #                 )
    #                 rot = carla.Rotation(pitch=0, yaw=float(yaws_deg[idx]), roll=0)

    #                 # Create bounding box
    #                 e = extents[idx]
    #                 bbox = carla.BoundingBox(loc, e)
    #                 bbox.rotation = rot

    #                 predicted_bounding_boxes[v_data.id].append(bbox)

    #     return predicted_bounding_boxes

    def forecast_vehicle_bbs_array(
        self,
        all_vehicle_data : List[VehicleDataEntry],
        prediction_horizons : Dict[int, int],
        default_future_frames : int
    ) -> Tuple[Dict[int, List[carla.BoundingBox]], Dict[int, List[carla.BoundingBox]]]:
        predicted_bounding_boxes = {v.id : [] for v in all_vehicle_data}
        predicted_dilated_bounding_boxes = {v.id : [] for v in all_vehicle_data}

        # Setup initial vectorized states (control, velocity, location, heading)
        previous_actions = np.array([[v.steer, v.throttle, v.brake] for v in all_vehicle_data])
        velocities = np.array([v.speed for v in all_vehicle_data])
        locations = np.array([[v.x, v.y, v.z] for v in all_vehicle_data])
        headings = np.array([v.heading for v in all_vehicle_data])

        # Setup prediction horizons for masking
        active_horizons = np.array([prediction_horizons.get(v.id, default_future_frames) for v in all_vehicle_data])

        # Cache vehicle extents
        extents = [v.vehicle.bounding_box.extent for v in all_vehicle_data]

        # Get route points per vehicle
        route_pts_by_idx : List[np.ndarray] = [v.vehicle_route_points for v in all_vehicle_data]

        # TODO: CLEAN THIS UP LATER, ESP THE HARDCODED DISTANCE CHECK
        def should_use_controller(
            route_pts_xy : np.ndarray,
            vehicle_pos_xy : np.ndarray,
        ) -> bool:
            if route_pts_xy.shape[0] < 2:
                return False

            end_pt = route_pts_xy[-1]
            prev_end_pt = route_pts_xy[-2]
            last_dir = end_pt - prev_end_pt
            n = max(1e-3, np.hypot(last_dir[0], last_dir[1]))

            # Convert to unit vector
            last_dir_unit = last_dir / n

            # Distance to end
            dx = end_pt[0] - vehicle_pos_xy[0]
            dy = end_pt[1] - vehicle_pos_xy[1]
            dist_to_end_sq = dx ** 2 + dy ** 2

            beyond_end = np.dot(vehicle_pos_xy - end_pt, last_dir_unit) > 0.0

            return (not beyond_end) and (dist_to_end_sq > 2.0 ** 2)

        # Forecast future locations, headings, velocities and create their bounding boxes
        base_front_buffer_m = self.config.front_buffer_m
        rear_buffer_m = self.config.rear_buffer_m
        lateral_buffer_m = self.config.lateral_buffer_m

        for i in range(default_future_frames):
            if i != 0:
                nearest_offsets = self._get_all_nearest_route_indices(locations, route_pts_by_idx, int(5)) # 1 point per 2m

                # Update steer using lateral controller where valid, otherwise use previous command
                for idx, v_data in enumerate(all_vehicle_data):
                    if i >= active_horizons[idx]:
                        continue

                    # Slice the route based on the vectorized search result
                    offset = nearest_offsets[idx]
                    route_pts_by_idx[idx] = route_pts_by_idx[idx][offset:]
                    current_route = route_pts_by_idx[idx]

                    if should_use_controller(current_route[:, :2], locations[idx, :2]):
                        # Pure pursuit
                        steer = self.lateral_controller.step_pure_pursuit(
                            route_points=current_route,
                            current_speed=velocities[idx],
                            vehicle_position=locations[idx],
                            vehicle_heading=headings[idx]
                        )
                        previous_actions[idx, 0] = steer

                # Vectorized Physics Update
                next_locs, next_heads, next_vels = self.vehicle_model.forecast_other_vehicles(
                    locations, headings, velocities, previous_actions
                )

                # Update vehicles that still have an active prediction horizon
                mask = i < active_horizons

                # Update states (if mask is False, keep previous state)
                locations[mask] = next_locs[mask]
                headings[mask] = next_heads[mask]
                velocities[mask] = next_vels[mask]

            # Vehicle predicted bounding boxes construction
            if i % self.steps_per_output == 0:
                yaws_deg = np.rad2deg(headings)

                for idx, v_data in enumerate(all_vehicle_data):
                    front_buffer_m = base_front_buffer_m
                    if v_data.traffic_type != "leading":
                        d_brake = self._get_braking_distance(
                            speed=velocities[idx],
                            reaction_time=self.config.min_agent_reaction_time_s,
                            brake_accel=self.config.brake_acceleration
                        )
                        front_buffer_m += d_brake

                    loc = carla.Location(
                        x=float(locations[idx, 0]),
                        y=float(locations[idx, 1]),
                        z=float(locations[idx, 2]),
                    )
                    rot = carla.Rotation(pitch=0, yaw=float(yaws_deg[idx]), roll=0)

                    # Create bounding box
                    e = extents[idx]
                    bbox = carla.BoundingBox(loc, e)
                    bbox.rotation = rot

                    # Dilate true bounding box to enforce minimum gaps
                    new_bbox = self._dilate_bbox(
                        bbox=bbox,
                        front_buffer_m=base_front_buffer_m,
                        rear_buffer_m=rear_buffer_m,
                        lateral_buffer_m=lateral_buffer_m,
                    )

                    # Dilate true bounding box with braking distance for soft costs
                    dilated_bbox = self._dilate_bbox(
                        bbox=bbox,
                        front_buffer_m=front_buffer_m,
                        rear_buffer_m=rear_buffer_m,
                        lateral_buffer_m=lateral_buffer_m,
                    )

                    predicted_bounding_boxes[v_data.id].append(new_bbox)
                    predicted_dilated_bounding_boxes[v_data.id].append(dilated_bbox)

        return predicted_bounding_boxes, predicted_dilated_bounding_boxes

    # TODO: PASS IN EGO VEHICLE DATA HERE
    def forecast_ego_vehicle_bbs(
        self,
        vehicle : carla.Vehicle,
        vehicle_route_points : np.ndarray,
        num_future_frames : int,
        target_speed : float = None,
        velocity_profile : Optional[np.ndarray] = None
    ) -> List[carla.BoundingBox]:
        self.lateral_controller.reset_state()

        vehicle_location = np.array(
            [vehicle.get_location().x, vehicle.get_location().y, vehicle.get_location().z]
        )
        vehicle_heading_angle = np.array([np.deg2rad(vehicle.get_transform().rotation.yaw)])
        vehicle_speed = np.array([vehicle.get_velocity().length()])

        has_velocity_profile = velocity_profile is not None and len(velocity_profile) > 0
        if has_velocity_profile:
            vehicle_target_speed = np.array([velocity_profile[0]])
        elif target_speed is not None:
            vehicle_target_speed = np.array([target_speed])
        else:
            vehicle_target_speed = vehicle_speed

        # Calculate throttle and brake command based on the target speed and current speed
        throttle, brake = self.long_controller.get_throttle_and_brake(
            hazard_brake=False,
            target_speed=vehicle_target_speed.item(),
            current_speed=vehicle_speed.item()
        )
        brake = float(brake)
        steering = self.lateral_controller.step(vehicle_route_points, vehicle_speed, vehicle_location, vehicle_heading_angle.item())

        action = np.array([steering, throttle, brake]).flatten()

        # Iterate over the future frames and forecast the agent's state
        future_bounding_boxes = []
        route_index = 0
        for i in range(num_future_frames):
            if has_velocity_profile:
                target_idx = min(i, len(velocity_profile) - 1)
                vehicle_target_speed = np.array([float(velocity_profile[target_idx])])

            # Forecast the next state using the kinematic bicycle model
            vehicle_location, vehicle_heading_angle, vehicle_speed = self.vehicle_model.forecast_ego_vehicle(vehicle_location, vehicle_heading_angle, vehicle_speed, action)

            # Update the route and extrapolate steering and throttle commands
            vehicle_forecast_route, route_index_offset = self._get_nearest_vehicle_route_point(vehicle_route_points, vehicle_location, route_index)
            route_index += route_index_offset

            steering = self.lateral_controller.step(vehicle_forecast_route, vehicle_speed, vehicle_location, vehicle_heading_angle.item())
            throttle, brake = self.long_controller.get_throttle_and_brake(hazard_brake=False, target_speed=vehicle_target_speed.item(), current_speed=vehicle_speed.item())
            brake = float(brake)

            action = np.array([steering, throttle, brake]).flatten()

            # Record predicted bounding boxes at output frequency
            if i % self.steps_per_output == 0:
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

        return future_bounding_boxes

    def forecast_ped_bbs(
        self,
        peds : List[carla.Walker],
        num_future_frames : int
    ) -> Dict[int, List[carla.BoundingBox]]:
        forecasted_ped_bbs = {}

        pedestrian_locations = np.array(
            [[ped.get_location().x, ped.get_location().y, ped.get_location().z] for ped in peds])
        pedestrian_speeds = np.array([ped.get_velocity().length() for ped in peds])
        # pedestrian_speeds = np.maximum(pedestrian_speeds, self.config.min_walker_speed)
        pedestrian_directions = np.array(
            [[ped.get_control().direction.x,
            ped.get_control().direction.y,
            ped.get_control().direction.z] for ped in peds])

        # Calculate future pedestrian locations based on their current locations, speeds, and directions
        future_pedestrian_locations = pedestrian_locations[:, None, :] + np.arange(1, num_future_frames + 1)[
            None, :, None] * pedestrian_directions[:, None, :] * pedestrian_speeds[:, None,
                                                                                None] / self.config.bicycle_frame_rate

        # Iterate over pedestrians and calculate their future bounding boxes
        for i, ped in enumerate(peds):
            bb, transform = ped.bounding_box, ped.get_transform()
            rotation = carla.Rotation(pitch=bb.rotation.pitch + transform.rotation.pitch,
                                        yaw=bb.rotation.yaw + transform.rotation.yaw,
                                        roll=bb.rotation.roll + transform.rotation.roll)
            extent = bb.extent
            extent.x = max(self.config.pedestrian_minimum_extent, extent.x)  # Ensure a minimum width
            extent.y = max(self.config.pedestrian_minimum_extent, extent.y)  # Ensure a minimum length

            pedestrian_future_bboxes = []
            for j in range(num_future_frames):
                if j % self.steps_per_output == 0:
                    location = carla.Location(future_pedestrian_locations[i, j, 0], future_pedestrian_locations[i, j, 1],
                                            future_pedestrian_locations[i, j, 2])

                    bounding_box = carla.BoundingBox(location, extent)
                    bounding_box.rotation = rotation
                    pedestrian_future_bboxes.append(bounding_box)

            forecasted_ped_bbs[ped.id] = pedestrian_future_bboxes

        return forecasted_ped_bbs

    @staticmethod
    def forecast_route_lane_bbs(
        route_points : np.ndarray,
        route_yaws : np.ndarray,
        lane_width : float,
        from_index : int,
        to_index : int,
        *,
        stride : int,
        spacing_m : int = 2,
        buffer_lat : float = 0.0,
        buffer_lon : float = 0.5,
    ) -> List[carla.BoundingBox]:
        forecasted_lane_bbs : List[carla.BoundingBox] = []

        half_len = 0.5 * float(spacing_m) + float(buffer_lon)
        half_wid = 0.5 * float(lane_width) + float(buffer_lat)
        extent = carla.Vector3D(x=half_len, y=half_wid, z=1.0)

        for i in range(from_index, to_index, stride * spacing_m):
            pt = route_points[i]
            yaw_deg = route_yaws[i]

            center = carla.Location(x=pt[0], y=pt[1], z=pt[2])

            bb = carla.BoundingBox(center, extent)
            bb.rotation = carla.Rotation(pitch=0, yaw=yaw_deg, roll=0)

            forecasted_lane_bbs.append(bb)

        return forecasted_lane_bbs
