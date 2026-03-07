import numpy as np
import carla

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from privileged_route_planner import PlannerState

@dataclass(frozen=True, slots=True)
class EgoVehicleData:
    """Ego vehicle information."""
    ego_vehicle : carla.Vehicle
    speed: float
    orientation: float
    position: List[float]

class EgoVehicleDataExtractor:
    """
    Extracts ego vehicle information from simulation context.

    Responsibilities:
    - Extract ego vehicle state (speed, position, orientation)
    - Process lane change information
    - Handle route and waypoint data
    """

    def __init__(self, config):
        self.config = config

    # -------------------------------------------------------------------- #
    #  Utility
    # -------------------------------------------------------------------- #

    def extract_ego_data(
        self,
        ego_vehicle : carla.Vehicle,
        planner_state: PlannerState
    ) -> EgoVehicleData:
        """
        Extract structured ego vehicle data from context.

        Args:
            ego_context: Dictionary containing ego vehicle information

        Returns:
            EgoVehicleData object with structured information
        """
        ego_transform = ego_vehicle.get_transform()
        ego_position = [ego_transform.location.x, ego_transform.location.y]
        ego_position = np.around(ego_position, 2).tolist()

        ego_orientation = round(np.deg2rad(ego_transform.rotation.yaw), 2)

        ego_speed = round(ego_vehicle.get_velocity().length(), 2)

        return EgoVehicleData(
            ego_vehicle=ego_vehicle,
            speed=ego_speed,
            orientation=ego_orientation,
            position=ego_position,
        )