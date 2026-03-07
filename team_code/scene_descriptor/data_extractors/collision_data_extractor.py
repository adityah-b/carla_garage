import carla
import numpy as np

from queue import Queue, Empty

from typing import List, Dict, Optional, Tuple, Literal, Union, Any
from dataclasses import dataclass, field

from config import GlobalConfig
from .vehicle_data_extractor import VehicleDataEntry, VehicleData
from .ped_data_extractor import PedestrianData
from .obstacle_data_extractor import ObstacleData

@dataclass
class CollisionData:
    # Actor information
    actor : Union[VehicleDataEntry, PedestrianData, ObstacleData, carla.Actor]

    # Collision data
    timestamp : float
    intensity : float

    @property
    def id(self) -> int:
        return self.actor.id

class CollisionDataExtractor:
    def __init__(self, config : GlobalConfig):
        self.config = config

    def extract_data(
        self,
        collision_sensor_data : Optional[Dict[str, Any]],
        vehicle_data : VehicleData,
        ped_data : PedestrianData,
        obs_data : ObstacleData,
    ) -> List[CollisionData]:
        if collision_sensor_data is None:
            return []

        # TODO: ADD SEARCH FOR COLLISION ACTOR DATA TYPE

        return [
            CollisionData(
                actor=collision_sensor_data["actor"],
                timestamp=collision_sensor_data["timestamp"],
                intensity=collision_sensor_data["intensity"],
            )
        ]
