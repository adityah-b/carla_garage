import numpy as np
import carla

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .base_actor_extractor import BaseActorExtractor
from .road_handler import RoadHandler, LaneVehicles, Lanelet
from privileged_route_planner import PlannerState

@dataclass(frozen=True, slots=True)
class VehicleData:
    """
    Structured vehicle data.
    """
    vehicle : carla.Vehicle
    id: int
    speed: float
    relative_orientation: float
    relative_position: Tuple[float]
    relative_distance: float

@dataclass(frozen=True, slots=True)
class LaneVehicleData:
    lanelet : Lanelet
    vehicle_data : List[VehicleData]

class VehicleDataExtractor(BaseActorExtractor):
    """
    Extracts and processes vehicle data from CARLA simulation.

    Responsibilities:
    - Convert vehicle actors to structured data
    - Calculate relative positions and orientations
    - Handle coordinate transformations
    """

    def __init__(self, config, carla_map : carla.Map):
        self.road_handler = RoadHandler(config, carla_map)

    def extract_vehicle_data(
        self,
        ego_wp : carla.Waypoint,
        planner_state : PlannerState,
        vehicles : List[carla.Vehicle],
    ) -> Dict[str, Dict[str, List[LaneVehicleData]]]:
        leading_vehicles_group = self.road_handler.get_leading_vehicles(ego_wp, planner_state, vehicles)
        trailing_vehicles_group = self.road_handler.get_trailing_vehicles(ego_wp, planner_state, vehicles)
        oncoming_vehicles_group = self.road_handler.get_oncoming_vehicles(ego_wp, planner_state, vehicles)
        cross_vehicles_group = self.road_handler.get_cross_vehicles(ego_wp, planner_state, vehicles)

        all_vehicles = {
            "leading" : self._group_vehicles(ego_wp, leading_vehicles_group),
            "trailing" : self._group_vehicles(ego_wp, trailing_vehicles_group),
            "oncoming" : self._group_vehicles(ego_wp, oncoming_vehicles_group),
            "cross" : self._group_vehicles(ego_wp, cross_vehicles_group),
        }

        return all_vehicles

    # -------------------------------------------------------------------- #
    #  Utility
    # -------------------------------------------------------------------- #

    def _group_vehicles(
        self,
        ego_wp : carla.Waypoint,
        vehicles_dict : Dict[str, List[LaneVehicles]]
    ) -> Dict[str, List[LaneVehicleData]]:
        """
        Extract vehicle data and group them based on their lane info
        """
        grouped_data = {}
        for lane_name, lane_vehicles_list in vehicles_dict.items():
            veh_data_list : List[LaneVehicleData] = []
            for lv in lane_vehicles_list:
                ll = lv.lanelet
                vehicles = lv.vehicles

                veh_data = self._extract_vehicle_data(ego_wp, vehicles)

                lv_data = LaneVehicleData(lanelet=ll, vehicle_data=veh_data)

                veh_data_list.append(lv_data)

            grouped_data[lane_name] = veh_data_list

        return grouped_data

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
            rel_pos = relative_positions[i][:2].round(2)
            data = VehicleData(
                vehicle=vehicles[i],
                id=int(vehicle_ids[i]),
                speed=round(float(speeds[i]), 2),
                relative_orientation=round(float(relative_yaws[i]), 2),
                relative_position=tuple(rel_pos),
                relative_distance=round(float(relative_distances[i]), 2)
            )
            vehicle_data.append(data)

        vehicle_data.sort(key=lambda v: (v.relative_distance))
        return vehicle_data
