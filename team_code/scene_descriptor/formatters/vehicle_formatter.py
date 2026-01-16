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
        any_added = False

        def dump_text(
            title: str,
            group: Dict[str, List[LaneVehicleData]]
        ) -> List[str]:
            group_text = []
            if not group:
                return group_text

            group_text.append(f"\t{title}:")
            any_added = False

            for lane_name, lv_list in group.items():
                if not lv_list:
                    continue

                lane_header_index = len(group_text)
                group_text.append(f"\t\t{lane_name}:")
                lane_count = 0

                for lv in lv_list:
                    for v in lv.vehicle_data:
                        # Ignore stationary trailing/oncoming/crossing vehicles
                        # TODO: FIX THIS SHIT
                        # if title in ["Cross Traffic", "Oncoming Traffic", "Trailing Traffic"] and v.speed <= 0.1:
                        #     continue

                        # Ignore trailing vehicles in ego lane (except cyclists)
                        is_cyclist = ("base_type" in v.vehicle.attributes and v.vehicle.attributes["base_type"] == "bicycle")
                        is_emergency = ("special_type" in v.vehicle.attributes and v.vehicle.attributes["special_type"] == "emergency")
                        if title == 'Trailing Traffic' and lane_name == 'ego' and not is_cyclist and not is_emergency:
                            continue

                        group_text.append(cls._format_vehicle_data(v, indent="\t\t\t", precision=precision))
                        lane_count += 1
                        any_added = True

                # Remove lane name if no actors present
                if lane_count == 0:
                    group_text.pop(lane_header_index)

            # Remove entry if nothing was added after title
            if not any_added:
                group_text = []

            return group_text

        leading_text = dump_text("Leading Traffic", grouped_vehicles.get("leading", {}))
        trailing_text = dump_text("Trailing Traffic", grouped_vehicles.get("trailing", {}))
        oncoming_text = dump_text("Oncoming Traffic", grouped_vehicles.get("oncoming", {}))
        cross_text = dump_text("Cross Traffic", grouped_vehicles.get("cross", {}))

        for text in [leading_text, trailing_text, oncoming_text, cross_text]:
            if text:
                any_added = True
                lines.extend(text)

        if not any_added:
            lines = []

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

        if "special_type" in vehicle_data.vehicle.attributes and vehicle_data.vehicle.attributes["special_type"] == "emergency":
            actor_type = "Emergency Vehicle"

        # Threshold below which we don't display speed
        SPEED_DISPLAY_THRESHOLD = 0.5

        parts = [f"{indent}{actor_type} ID: {vehicle_data.id}"]

        # Only include speed if above threshold
        if vehicle_data.speed >= SPEED_DISPLAY_THRESHOLD:
            parts.append(f"Speed: {f(vehicle_data.speed, precision)}")

        # f"Relative Position: {f(vehicle_data.relative_position, precision)}, "
        # f"Relative Orientation: {f(vehicle_data.relative_orientation, precision)}, "
        parts.append(f"Distance: {f(vehicle_data.relative_distance, precision)}")
        # f"Distance: {f(vehicle_data.relative_distance, precision)}, "
        # f"Intruding Ego Lane: {vehicle_data.is_intruding}"

        return ", ".join(parts)
