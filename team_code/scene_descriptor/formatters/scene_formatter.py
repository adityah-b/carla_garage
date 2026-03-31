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

        formatted_text.append(VehicleFormatter.format_vehicles(vehicle_data=scene_data.vehicle_data, precision=precision))

        if scene_data.ped_data:
            formatted_text.append(PedestrianFormatter.format_pedestrians(peds=scene_data.ped_data, precision=precision))

        formatted_text.append(ObstacleFormatter.format_obstacles(obstacle_data=scene_data.obstacle_data, precision=precision))

        formatted_text.append(RouteFormatter.format_route(route_data=scene_data.route_data, precision=precision))

        return "\n".join(formatted_text)

    def summarize(
        self,
        scene_data: SceneData,
    ) -> str:
        """
        Sectioned natural-language summary for LLM/VLM consumption.
        Order:
          1. Route intent
          2. Immediate control
          3. Ego state
          4. Vulnerable road users
          5. Obstacles / blockages
          6. Dynamic conflicts
        """
        sections = []

        if scene_data.route_data:
            text = RouteFormatter.summarize(scene_data.route_data)
            if text:
                sections.append(text)

        if scene_data.traffic_data:
            text = TrafficFormatter.summarize(scene_data.traffic_data)
            if text:
                sections.append(text)

        if scene_data.ego_data:
            text = EgoVehicleFormatter.summarize(scene_data.ego_data)
            if text:
                sections.append(text)

        if scene_data.ped_data:
            text = PedestrianFormatter.summarize(scene_data.ped_data)
            if text:
                sections.append(text)

        if scene_data.obstacle_data:
            text = ObstacleFormatter.summarize(scene_data.obstacle_data)
            if text:
                sections.append(text)

        if scene_data.vehicle_data:
            text = VehicleFormatter.summarize(scene_data.vehicle_data)
            if text:
                sections.append(text)

        return "\n\n".join(sections)
