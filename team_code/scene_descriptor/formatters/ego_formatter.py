from typing import Dict, List
from .base_formatter import BaseFormatter
from scene_descriptor.data_extractors.ego_data_extractor import EgoVehicleData

class EgoVehicleFormatter(BaseFormatter):
    @classmethod
    def format_ego(
        cls,
        ego_vehicle_data: EgoVehicleData,
        precision: int = 2
    ) -> str:
        f = cls.fmt
        lines : List[str] = ["Ego Data:"]

        text_data = (
            f"\tSpeed: {f(ego_vehicle_data.speed, precision=precision)}, "
            f"Orientation: {f(ego_vehicle_data.orientation, precision=precision)}, "
            f"Position: {f(ego_vehicle_data.position, precision=precision)}"
        )

        lines.append(text_data)

        return "\n".join(lines)
