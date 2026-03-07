import carla

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from config import GlobalConfig
from privileged_route_planner import PlannerState

from .traffic_data_extractor import TrafficData, TrafficDataExtractor
from .ego_data_extractor import EgoVehicleData, EgoVehicleDataExtractor
from .vehicle_data_extractor import VehicleData, VehicleDataExtractor
from .ped_data_extractor import PedestrianData, PedestrianDataExtractor
from .obstacle_data_extractor import ObstacleData, ObstacleDataExtractor
from .route_extractor import RouteData, RouteDataExtractor
from .collision_data_extractor import CollisionData, CollisionDataExtractor

@dataclass(frozen=True, slots=True)
class SceneData:
    traffic_data : Optional[TrafficData]
    ego_data : Optional[EgoVehicleData]
    vehicle_data : VehicleData
    ped_data : Optional[List[PedestrianData]]
    obstacle_data : ObstacleData
    route_data : Optional[RouteData]
    collision_data : List[CollisionData]

class SceneExtractor:
    def __init__(self, config : GlobalConfig, carla_map : carla.Map):
        self.config = config
        self.carla_map = carla_map

        self._traffic_extractor = TrafficDataExtractor(self.config)
        self._ego_extractor = EgoVehicleDataExtractor(self.config)
        self._vehicle_extractor = VehicleDataExtractor(self.config, self.carla_map)
        self._ped_extractor = PedestrianDataExtractor(self.config, self.carla_map)
        self._obstacle_extractor = ObstacleDataExtractor(self.config, self.carla_map)
        self._route_extractor = RouteDataExtractor(self.config)
        self._collision_extractor = CollisionDataExtractor(self.config)

    def extract_scene(
        self,
        ego_vehicle : carla.Vehicle,
        actors : carla.ActorList,
        planner_state : PlannerState,
        lidar_data : Dict,
        collision_sensor_data : Dict,
    ) -> SceneData:
        ego_wp = self.carla_map.get_waypoint(ego_vehicle.get_location(), lane_type=carla.LaneType.Any)
        ego_transform = ego_vehicle.get_transform()
        ego_loc = ego_vehicle.get_location()

        # Extract structured data using specialized extractors
        traffic_data = self._traffic_extractor.extract_traffic_data(ego_transform=ego_transform, planner_state=planner_state)
        ego_data = self._ego_extractor.extract_ego_data(ego_vehicle=ego_vehicle, planner_state=planner_state)

        npc_vehicles = [
            veh for veh in actors.filter("*vehicle*")
            if veh.id != ego_vehicle.id
        ]
        vehicle_data = self._vehicle_extractor.extract_data(
            ego_transform=ego_transform,
            ego_wp=ego_wp,
            planner_state=planner_state,
            vehicles=npc_vehicles
        )

        pedestrians = [
            ped for ped in actors.filter("*walker*")
            if ped.get_location().distance(ego_loc) < self.config.detection_radius
        ]
        ped_data = self._ped_extractor.extract_ped_data(ego_transform=ego_transform, peds=pedestrians)

        static_obstacles = list(actors.filter("static.*"))
        # Check if actor is dynamically spawned non-moving vehicle (treat as static object)
        for vehicle in npc_vehicles:
            veh_control = vehicle.get_control()
            veh_type = vehicle.attributes.get("base_type", "")
            if veh_control.hand_brake and veh_type != "bicycle":
                static_obstacles.append(vehicle)

        obstacle_data = self._obstacle_extractor.extract_obstacle_data(
            ego_transform=ego_transform,
            ego_wp=ego_wp, obstacles=static_obstacles, planner_state=planner_state, lidar_data=lidar_data
        )

        route_data = self._route_extractor.extract_route_data(ego_vehicle, ego_wp, planner_state)

        collision_data = self._collision_extractor.extract_data(
            collision_sensor_data=collision_sensor_data,
            vehicle_data=vehicle_data,
            ped_data=ped_data,
            obs_data=obstacle_data
        )

        return SceneData(
            traffic_data=traffic_data if traffic_data else None,
            ego_data=ego_data if ego_data else None,
            vehicle_data=vehicle_data,
            ped_data=ped_data if ped_data else None,
            obstacle_data=obstacle_data,
            route_data=route_data if route_data else None,
            collision_data=collision_data
        )
