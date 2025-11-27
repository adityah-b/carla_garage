import carla
import numpy as np

from typing import List, Tuple, Dict

from config import GlobalConfig
from kinematic_bicycle_model import KinematicBicycleModel
from lateral_controller import LateralPIDController
from longitudinal_controller import LongitudinalLinearRegressionController

class MotionForecaster:
    def __init__(self, config : GlobalConfig):
        self.config = config
        self.vehicle_model = KinematicBicycleModel(self.config)

        self.lateral_controller = LateralPIDController(self.config)
        self.long_controller = LongitudinalLinearRegressionController(self.config)

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

    def forecast_vehicle_bbs_array(
        self,
        nearby_actors : List[carla.Vehicle],
        num_future_frames : int
    ) -> Dict[int, List[carla.BoundingBox]]:
        predicted_bounding_boxes = {}

        previous_controls = [actor.get_control() for actor in nearby_actors]
        previous_actions = np.array([[control.steer, control.throttle, control.brake] for control in previous_controls])

        # Get the current velocities, locations, and headings of the nearby actors
        velocities = np.array([actor.get_velocity().length() for actor in nearby_actors])
        locations = np.array([[actor.get_location().x,
                               actor.get_location().y,
                               actor.get_location().z] for actor in nearby_actors])
        headings = np.deg2rad(np.array([actor.get_transform().rotation.yaw for actor in nearby_actors]))

        # Initialize arrays to store future locations, headings, and velocities
        future_locations = np.empty((num_future_frames, len(nearby_actors), 3), dtype="float")
        future_headings = np.empty((num_future_frames, len(nearby_actors)), dtype="float")
        future_velocities = np.empty((num_future_frames, len(nearby_actors)), dtype="float")

        # Forecast the future locations, headings, and velocities for the nearby actors
        for i in range(num_future_frames):
            locations, headings, velocities = self.vehicle_model.forecast_other_vehicles(
                locations, headings, velocities, previous_actions)
            future_locations[i] = locations.copy()
            future_velocities[i] = velocities.copy()
            future_headings[i] = headings.copy()

        # Convert future headings to degrees
        future_headings = np.rad2deg(future_headings)

        # Calculate the predicted bounding boxes for each nearby actor and future frame
        for actor_idx, actor in enumerate(nearby_actors):
            predicted_actor_boxes = []

            for i in range(num_future_frames):
                # Calculate the future location of the actor
                location = carla.Location(x=future_locations[i, actor_idx, 0].item(),
                                        y=future_locations[i, actor_idx, 1].item(),
                                        z=future_locations[i, actor_idx, 2].item())

                # Calculate the future rotation of the actor
                rotation = carla.Rotation(pitch=0, yaw=future_headings[i, actor_idx], roll=0)

                # Get the extent (dimensions) of the actor's bounding box
                extent = actor.bounding_box.extent
                # Otherwise we would increase the extent of the bounding box of the vehicle
                extent = carla.Vector3D(x=extent.x, y=extent.y, z=extent.z)

                # Adjust the bounding box size based on velocity and lane change maneuver to adjust for
                # uncertainty during forecasting
                s = self.config.high_speed_min_extent_x_other_vehicle
                extent.x *= self.config.slow_speed_extent_factor_ego if future_velocities[
                    i, actor_idx] < self.config.extent_other_vehicles_bbs_speed_threshold else max(
                        s,
                        self.config.high_speed_min_extent_x_other_vehicle * float(i) / float(num_future_frames))
                extent.y *= self.config.slow_speed_extent_factor_ego if future_velocities[
                    i, actor_idx] < self.config.extent_other_vehicles_bbs_speed_threshold else max(
                        self.config.high_speed_min_extent_y_other_vehicle,
                        self.config.high_speed_extent_y_factor_other_vehicle * float(i) / float(num_future_frames))

                # Create the bounding box for the future frame
                bounding_box = carla.BoundingBox(location, extent)
                bounding_box.rotation = rotation

                # Append the bounding box to the list of predicted bounding boxes for this actor
                predicted_actor_boxes.append(bounding_box)

            # Store the predicted bounding boxes for this actor in the dictionary
            predicted_bounding_boxes[actor.id] = predicted_actor_boxes

        return predicted_bounding_boxes

    def forecast_vehicle_bbs(
        self,
        vehicle : carla.Vehicle,
        vehicle_route_points : np.ndarray,
        num_future_frames : int,
        target_speed : float = None,
    ) -> List[carla.BoundingBox]:
        self.lateral_controller.reset_state()

        vehicle_location = np.array(
            [vehicle.get_location().x, vehicle.get_location().y, vehicle.get_location().z]
        )
        vehicle_heading_angle = np.array([np.deg2rad(vehicle.get_transform().rotation.yaw)])
        vehicle_speed = np.array([vehicle.get_velocity().length()])

        vehicle_target_speed = np.array([target_speed]) if target_speed is not None else vehicle_speed

        # Calculate the throttle command based on the target speed and current speed
        throttle = self.long_controller.get_throttle_extrapolation(vehicle_target_speed, vehicle_speed)
        steering = self.lateral_controller.step(vehicle_route_points, vehicle_speed, vehicle_location, vehicle_heading_angle.item())
        action = np.array([steering, throttle, 0.0]).flatten()

        # Iterate over the future frames and forecast the agent's state
        future_bounding_boxes = []
        route_index = 0
        for i in range(num_future_frames):
            # Forecast the next state using the kinematic bicycle model
            vehicle_location, vehicle_heading_angle, vehicle_speed = self.vehicle_model.forecast_ego_vehicle(vehicle_location, vehicle_heading_angle, vehicle_speed, action)

            # Update the route and extrapolate steering and throttle commands
            vehicle_forecast_route, route_index_offset = self._get_nearest_vehicle_route_point(vehicle_route_points, vehicle_location, route_index)
            route_index += route_index_offset

            steering = self.lateral_controller.step(vehicle_forecast_route, vehicle_speed, vehicle_location, vehicle_heading_angle.item())
            throttle = self.long_controller.get_throttle_extrapolation(vehicle_target_speed, vehicle_speed)
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
        pedestrian_speeds = np.maximum(pedestrian_speeds, self.config.min_walker_speed)
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
                location = carla.Location(future_pedestrian_locations[i, j, 0], future_pedestrian_locations[i, j, 1],
                                        future_pedestrian_locations[i, j, 2])

                bounding_box = carla.BoundingBox(location, extent)
                bounding_box.rotation = rotation
                pedestrian_future_bboxes.append(bounding_box)

            forecasted_ped_bbs[ped.id] = pedestrian_future_bboxes

        return forecasted_ped_bbs
