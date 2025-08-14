from typing import Dict, List, Tuple, Any, Optional

from .ego_formatter import EgoVehicleFormatter, EgoVehicleData
from .vehicle_formatter import VehicleFormatter, LaneVehicleData
from .traffic_formatter import TrafficFormatter, TrafficData
from .ped_formatter import PedestrianFormatter, PedestrianData

class SceneFormatter:
    def format_scene(
        self,
        traffic : Optional[TrafficData],
        ego : Optional[EgoVehicleData],
        vehicles : Optional[Dict[str, Dict[str, List[LaneVehicleData]]]],
        peds : Optional[PedestrianData],
        precision : int = 2
    ) -> str:
        formatted_text : List[str] = []
        if traffic:
            formatted_text.append(TrafficFormatter.format_traffic(traffic_data=traffic, precision=precision))
        if ego:
            formatted_text.append(EgoVehicleFormatter.format_ego(ego_vehicle_data=ego, precision=precision))
        if vehicles:
            formatted_text.append(VehicleFormatter.format_vehicles(grouped_vehicles=vehicles, precision=precision))
        if peds:
            formatted_text.append(PedestrianFormatter.format_pedestrians(peds=peds, precision=precision))

        return "\n".join(formatted_text)
