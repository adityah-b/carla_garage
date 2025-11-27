from scene_descriptor.data_extractors.scene_extractor import SceneData

from .ego_formatter import EgoVehicleFormatter
from .vehicle_formatter import VehicleFormatter
from .traffic_formatter import TrafficFormatter
from .ped_formatter import PedestrianFormatter
from .obstacle_formatter import ObstacleFormatter
from .route_formatter import RouteFormatter

class SceneFormatter:
    def format_scene(
        self,
        scene_data : SceneData,
        precision : int = 2
    ) -> str:
        formatted_text = []
        if scene_data.traffic_data:
            formatted_text.append(TrafficFormatter.format_traffic(traffic_data=scene_data.traffic_data, precision=precision))
        if scene_data.ego_data:
            formatted_text.append(EgoVehicleFormatter.format_ego(ego_vehicle_data=scene_data.ego_data, precision=precision))
        if scene_data.vehicle_data:
            formatted_text.append(VehicleFormatter.format_vehicles(grouped_vehicles=scene_data.vehicle_data, precision=precision))
        if scene_data.ped_data:
            formatted_text.append(PedestrianFormatter.format_pedestrians(peds=scene_data.ped_data, precision=precision))
        if scene_data.obstacle_data:
            formatted_text.append(ObstacleFormatter.format_obstacles(obstacles=scene_data.obstacle_data, precision=precision))
        if scene_data.route_data:
            formatted_text.append(RouteFormatter.format_route(route_data=scene_data.route_data, precision=precision))

        return "\n".join(formatted_text)
