import numpy as np
import carla

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from road_handler import RoadHandler, LaneVehicles, Lanelet
from privileged_route_planner import PlannerState

@dataclass(frozen=True, slots=True)
class VehicleData:
    """
    Structured vehicle data.
    """
    vehicle_id: int
    speed: float
    relative_orientation: float
    relative_position: List[float]
    relative_distance: float

@dataclass(frozen=True, slots=True)
class LaneVehicleData:
    lanelet : Lanelet
    vehicle_data : List[VehicleData]

class VehicleDataExtractor:
    """
    Extracts and processes vehicle data from CARLA simulation.

    Responsibilities:
    - Convert vehicle actors to structured data
    - Calculate relative positions and orientations
    - Handle coordinate transformations
    """

    def __init__(self, config, carla_map : carla.Map):
        self.road_handler = RoadHandler(config, carla_map)

    def get_all_vehicle_traffic(
        self,
        ego_wp : carla.Waypoint,
        planner_state : PlannerState,
        vehicles : List[carla.Vehicle],
    ) -> Dict[str, Dict[str, Any]]:
        leading_vehicles_group = self.road_handler.get_leading_vehicles(planner_state, vehicles)
        # trailing_vehicles_group = self.road_handler.get_trailing_vehicles(planner_state, vehicles)
        # oncoming_vehicles_group = self.road_handler.get_oncoming_vehicles(planner_state, vehicles)

        for lane_name, lane_vehicles in leading_vehicles_group.items():
            # TODO: Might need to fix return type and return LaneVehicles instead of List[LaneVehicles]
            leading_vehicles = lane_vehicles[0].vehicles

            leading_data = self._extract_vehicle_data(ego_wp, leading_vehicles)

    # -------------------------------------------------------------------- #
    #  Utility
    # -------------------------------------------------------------------- #

    def _extract_vehicle_data(
        self,
        ego_wp: carla.Waypoint,
        vehicles: List[carla.Vehicle],
        closest_only: bool = False
    ) -> List[VehicleData]:
        if not vehicles:
            return []

        ego_transform_matrix = self._get_ego_transform_matrix(ego_wp)
        ego_yaw = self._get_ego_yaw(ego_wp)

        # Vectorized data extraction
        vehicle_data = self._extract_vectorized_data(vehicles, ego_transform_matrix, ego_yaw)

        if closest_only and vehicle_data:
            # Return only the closest vehicle
            closest_idx = np.argmin([data.relative_distance for data in vehicle_data])
            return [vehicle_data[closest_idx]]

        return vehicle_data

    # -------------------------------------------------------------------- #
    #  Private
    # -------------------------------------------------------------------- #

    def _get_ego_transform_matrix(self, ego_wp: carla.Waypoint) -> np.ndarray:
        """
        Extract ego vehicle transformation matrix.
        """
        return np.array(ego_wp.transform.get_matrix())

    def _get_ego_yaw(self, ego_wp: carla.Waypoint) -> float:
        """
        Extract ego vehicle yaw angle in radians.
        """
        return np.deg2rad(ego_wp.transform.rotation.yaw)

    def _extract_vectorized_data(
        self,
        vehicles: List[carla.Vehicle],
        ego_matrix: np.ndarray,
        ego_yaw: float
    ) -> List[VehicleData]:
        """
        Extract vehicle data using vectorized operations for efficiency.

        Args:
            vehicles: List of CARLA vehicle actors
            ego_matrix: Ego vehicle transformation matrix
            ego_yaw: Ego vehicle yaw angle

        Returns:
            List of VehicleData objects
        """
        # Extract all data in vectorized form
        transform_matrices = np.array([v.get_transform().get_matrix() for v in vehicles])
        yaws = np.array([np.deg2rad(v.get_transform().rotation.yaw) for v in vehicles])
        speeds = np.array([v.get_velocity().length() for v in vehicles])
        vehicle_ids = np.array([v.id for v in vehicles])

        # Calculate relative positions
        relative_positions = self._calculate_relative_positions(transform_matrices, ego_matrix)

        # Calculate relative yaws
        relative_yaws = self._normalize_angles(yaws - ego_yaw)

        # Calculate distances
        relative_distances = np.linalg.norm(relative_positions, axis=1)

        # Create VehicleData objects
        vehicle_data = []
        for i in range(len(vehicles)):
            data = VehicleData(
                vehicle_id=int(vehicle_ids[i]),
                speed=round(float(speeds[i]), 2),
                relative_orientation=round(float(relative_yaws[i]), 2),
                relative_position=relative_positions[i][:2].round(2).tolist(),
                relative_distance=round(float(relative_distances[i]), 2)
            )
            vehicle_data.append(data)

        return vehicle_data

    def _calculate_relative_positions(
        self,
        vehicle_matrices: np.ndarray,
        ego_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate relative positions of vehicles with respect to ego vehicle.

        Args:
            vehicle_matrices: Nx4x4 array of vehicle transformation matrices
            ego_matrix: 4x4 ego vehicle transformation matrix

        Returns:
            Nx3 array of relative positions
        """
        # Get positions from transformation matrices
        vehicle_positions = vehicle_matrices[:, :3, 3]
        ego_position = ego_matrix[:3, 3]

        # Calculate relative positions in world coordinates
        relative_world = vehicle_positions - ego_position[np.newaxis, :]

        # Transform to ego vehicle coordinate system
        ego_rotation = ego_matrix[:3, :3]
        relative_ego = (ego_rotation.T @ relative_world.T).T

        return relative_ego

    def _normalize_angles(self, angles: np.ndarray) -> np.ndarray:
        """Normalize angles to [-π, π] range."""
        return (angles + np.pi) % (2 * np.pi) - np.pi
