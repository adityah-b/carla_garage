from typing import Dict, List
from .base_formatter import BaseFormatter
from scene_descriptor.data_extractors.vehicle_data_extractor import VehicleDataEntry, VehicleData

class VehicleFormatter(BaseFormatter):
    @classmethod
    def format_vehicles(
        cls,
        vehicle_data: VehicleData,
        precision: int = 2,
    ) -> str:
        lines: List[str] = ["Vehicle Data:"]
        any_added = False

        def dump_text(
            title: str,
            group: Dict[str, List[VehicleDataEntry]]
        ) -> List[str]:
            group_text = []
            if not group:
                return group_text

            group_text.append(f"\t{title}:")
            any_added = False

            for lane_name, all_vehicle_data in group.items():
                if not all_vehicle_data:
                    continue

                lane_header_index = len(group_text)
                group_text.append(f"\t\t{lane_name}:")
                lane_count = 0

                for v_entry in all_vehicle_data:
                    # Ignore stationary trailing/oncoming/crossing vehicles
                    # TODO: FIX THIS SHIT
                    # if title in ["Cross Traffic", "Oncoming Traffic", "Trailing Traffic"] and v.speed <= 0.1:
                    #     continue

                    # Ignore trailing vehicles in ego lane (except cyclists)
                    is_cyclist = v_entry.vehicle_type == "cyclist"
                    is_emergency_vehicle = v_entry.vehicle_type == "emergency"
                    if title == 'Trailing Traffic' and lane_name == 'ego' and not is_cyclist and not is_emergency_vehicle:
                        continue

                    group_text.append(cls._format_vehicle_data(v_entry, indent="\t\t\t", precision=precision))
                    lane_count += 1
                    any_added = True

                # Remove lane name if no actors present
                if lane_count == 0:
                    group_text.pop(lane_header_index)

            # Remove entry if nothing was added after title
            if not any_added:
                group_text = []

            return group_text

        leading_text = dump_text("Leading Traffic", vehicle_data.view.get("leading", {}))
        trailing_text = dump_text("Trailing Traffic", vehicle_data.view.get("trailing", {}))
        oncoming_text = dump_text("Oncoming Traffic", vehicle_data.view.get("oncoming", {}))
        crossing_text = dump_text("Crossing Traffic", vehicle_data.view.get("crossing", {}))

        for text in [leading_text, trailing_text, oncoming_text, crossing_text]:
            if text:
                any_added = True
                lines.extend(text)

        if not any_added:
            lines = []

        return "\n".join(lines)

    @classmethod
    def _format_vehicle_data(
        cls,
        vehicle_data_entry: VehicleDataEntry,
        indent: str = "",
        precision: int = 2,
    ) -> str:
        f = cls.fmt
        vehicle_type = "Vehicle"
        if vehicle_data_entry.vehicle_type == "cyclist":
            vehicle_type = "Cyclist"
        if vehicle_data_entry.vehicle_type == "emergency":
            # TODO: NEED TO FIGURE OUT A WAY TO DISTINGUISH BETWEEN ACTUAL AND FAKE EMERGENCIES
            # TODO: CURRENTLY ONLY TREATING EMERGENCY VEHICLES WITH ACTIVE SIRENS AS TRUE EMERGENCY VEHICLES
            if vehicle_data_entry.emergency_sirens_active:
                vehicle_type = "Emergency Vehicle"

        # Threshold below which we don't display speed
        SPEED_DISPLAY_THRESHOLD = 0.5

        parts = [f"{indent}{vehicle_type} ID: {vehicle_data_entry.id}"]

        # Only include speed if above threshold
        if vehicle_data_entry.speed >= SPEED_DISPLAY_THRESHOLD:
            parts.append(f"Speed: {f(vehicle_data_entry.speed, precision)}")

        # f"Relative Position: {f(vehicle_data.relative_position, precision)}, "
        # f"Relative Orientation: {f(vehicle_data.relative_orientation, precision)}, "
        parts.append(f"Distance: {f(vehicle_data_entry.relative_distance, precision)}")

        # if vehicle_data.intrusion_index >= 0:
        #     parts.append(f"Intruding Ego Lane")

        if vehicle_type == "Emergency Vehicle":
            if vehicle_data_entry.emergency_sirens_active:
                parts.append(f"Emergency Sirens Active")
        #     else:
        #         parts.append(f"Emergency Sirens Inactive")

        intrudes_ego = False
        if vehicle_data_entry.intrudes_ego:
            if vehicle_data_entry.traffic_type == "oncoming":
                intrudes_ego = True
            elif vehicle_data_entry.traffic_type == "leading" and vehicle_data_entry.vehicle_type == "cyclist":
                intrudes_ego = True

        if intrudes_ego:
            parts.append(f"Intruding Ego Lane")

        return ", ".join(parts)
