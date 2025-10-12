import carla

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from config import GlobalConfig
from privileged_route_planner import PlannerState

from .traffic_data_extractor import TrafficData, TrafficDataExtractor
from .ego_data_extractor import EgoVehicleData, EgoVehicleDataExtractor
from .vehicle_data_extractor import LaneVehicleData, VehicleDataExtractor
from .ped_data_extractor import PedestrianData, PedestrianDataExtractor
from .route_extractor import RouteData, RouteDataExtractor

@dataclass(frozen=True, slots=True)
class SceneData:
    traffic_data : Optional[TrafficData]
    ego_data : Optional[EgoVehicleData]
    vehicle_data : Optional[Dict[str, Dict[str, List[LaneVehicleData]]]]
    ped_data : Optional[List[PedestrianData]]
    route_data : Optional[RouteData]


class SceneExtractor:
    def __init__(self, config : GlobalConfig, carla_map : carla.Map):
        self.config = config
        self.carla_map = carla_map

        self._traffic_extractor = TrafficDataExtractor(self.config)
        self._ego_extractor = EgoVehicleDataExtractor(self.config)
        self._vehicle_extractor = VehicleDataExtractor(self.config, self.carla_map)
        self._ped_extractor = PedestrianDataExtractor(self.config, self.carla_map)
        self._route_extractor = RouteDataExtractor(self.config)

    def extract_scene(
        self,
        ego_vehicle : carla.Vehicle,
        actors : carla.ActorList,
        planner_state : PlannerState
    ) -> SceneData:
        ego_wp = self.carla_map.get_waypoint(ego_vehicle.get_location(), lane_type=carla.LaneType.Any)
        ego_loc = ego_vehicle.get_location()

        # Extract structured data using specialized extractors
        traffic_data = self._traffic_extractor.extract_traffic_data(planner_state=planner_state)
        ego_data = self._ego_extractor.extract_ego_data(ego_vehicle=ego_vehicle, planner_state=planner_state)

        npc_vehicles = [
            veh for veh in actors.filter("*vehicle*")
            if veh.id != ego_vehicle.id
        ]
        vehicle_data = self._vehicle_extractor.extract_vehicle_data(ego_wp=ego_wp, planner_state=planner_state, vehicles=npc_vehicles)

        pedestrians = [
            ped for ped in actors.filter("*walker*")
            if ped.get_location().distance(ego_loc) < self.config.detection_radius
        ]
        ped_data = self._ped_extractor.extract_ped_data(ego_wp=ego_wp, peds=pedestrians)

        route_data = self._route_extractor.extract_route_data(ego_vehicle, ego_wp, planner_state)

        return SceneData(
            traffic_data=traffic_data if traffic_data else None,
            ego_data=ego_data if ego_data else None,
            vehicle_data=vehicle_data if vehicle_data else None,
            ped_data=ped_data if ped_data else None,
            route_data=route_data if route_data else None,
        )
