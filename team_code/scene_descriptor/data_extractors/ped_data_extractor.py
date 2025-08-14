import carla
import numpy as np

from typing import List, Dict, Optional
from dataclasses import dataclass

from config import GlobalConfig

from .base_actor_extractor import BaseActorExtractor

@dataclass(frozen=True, slots=True)
class PedestrianData:
    pedestrian : carla.Walker
    id : int
    speed : float
    relative_orientation : float
    relative_position : List[float]
    relative_distance : float
    is_on_road : bool

class PedestrianDataExtractor(BaseActorExtractor):
    def __init__(self, config : GlobalConfig, carla_map : carla.Map):
        self.config = config
        self.carla_map = carla_map

    def extract_ped_data(
        self,
        ego_wp : carla.Waypoint,
        peds : List[carla.Walker]
    ) -> List[PedestrianData]:
        if not peds:
            return []

        ego_transform_matrix = self._get_ego_transform_matrix(ego_wp)
        ego_yaw = self._get_ego_yaw(ego_wp)

        ped_data = self._extract_vectorized_data(peds, ego_transform_matrix, ego_yaw)

        return ped_data

    def _extract_vectorized_data(
        self,
        peds : List[carla.Walker],
        ego_matrix : np.ndarray,
        ego_yaw : float
    ) -> List[PedestrianData]:
        # Extract all data in vectorized form
        transform_matrices = np.array([p.get_transform().get_matrix() for p in peds])
        yaws = np.array([np.deg2rad(p.get_transform().rotation.yaw) for p in peds])
        speeds = np.array([p.get_velocity().length() for p in peds])
        ped_ids = np.array([p.id for p in peds])

        # Calculate relative positions
        relative_positions = self._calculate_relative_positions(transform_matrices, ego_matrix)

        # Calculate relative yaws
        relative_yaws = self._normalize_angles(yaws - ego_yaw)

        # Calculate distances
        relative_distances = np.linalg.norm(relative_positions, axis=1)

        # Create PedestrianData objects
        ped_data = []
        for i in range(len(peds)):
            # Check if pedestrian is on road
            ped_wp = self.carla_map.get_waypoint(peds[i].get_location(), project_to_road=False, lane_type=carla.LaneType.Driving)
            is_on_road = True if ped_wp else False

            data = PedestrianData(
                pedestrian=peds[i],
                id=int(ped_ids[i]),
                speed=round(float(speeds[i]), 2),
                relative_orientation=round(float(relative_yaws[i]), 2),
                relative_position=relative_positions[i][:2].round(2).tolist(),
                relative_distance=round(float(relative_distances[i]), 2),
                is_on_road=is_on_road
            )
            ped_data.append(data)

        return ped_data
