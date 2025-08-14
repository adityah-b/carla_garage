from typing import Dict, List
from .base_formatter import BaseFormatter
from scene_descriptor.data_extractors.vehicle_data_extractor import VehicleData, LaneVehicleData

class VehicleFormatter(BaseFormatter):
    @classmethod
    def format_vehicles(
        cls,
        grouped_vehicles: Dict[str, Dict[str, List[LaneVehicleData]]],
        precision: int = 2,
    ) -> str:
        lines: List[str] = ["Vehicle Data:"]

        def dump_text(
            title: str,
            group: Dict[str, List[LaneVehicleData]]
        ) -> List[str]:
            group_text = []
            if group:
                group_text.append(f"\t{title}:")
                for lane_name, lv_list in group.items():
                    if lv_list:
                        group_text.append(f"\t\t{lane_name}:")
                        for lv in lv_list:
                            for v in lv.vehicle_data:
                                group_text.append(cls._format_vehicle_data(v, indent="\t\t\t", precision=precision))

                # Remove entry if nothing was added after title
                if len(group_text) <= 1:
                    group_text = []

            return group_text

        leading_text = dump_text("Leading Traffic", grouped_vehicles.get("leading", {}))
        trailing_text = dump_text("Trailing Traffic", grouped_vehicles.get("trailing", {}))
        oncoming_text = dump_text("Oncoming Traffic", grouped_vehicles.get("oncoming", {}))
        cross_text = dump_text("Cross Traffic", grouped_vehicles.get("cross", {}))

        for text in [leading_text, trailing_text, oncoming_text, cross_text]:
            if text:
                lines.extend(text)

        return "\n".join(lines)

    @classmethod
    def _format_vehicle_data(
        cls,
        vehicle_data: VehicleData,
        indent: str = "",
        precision: int = 2,
    ) -> str:
        f = cls.fmt
        actor_type = "Vehicle"
        if "base_type" in vehicle_data.vehicle.attributes and vehicle_data.vehicle.attributes["base_type"] == "bicycle":
            actor_type = "Cylist"

        return (
            f"{indent}{actor_type} ID: {vehicle_data.id}, "
            f"Speed: {f(vehicle_data.speed, precision)}, "
            f"Relative Position: {f(vehicle_data.relative_position, precision)}, "
            f"Relative Orientation: {f(vehicle_data.relative_orientation, precision)}, "
            f"Relative Distance: {f(vehicle_data.relative_distance, precision)}"
        )
