import math

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
                    is_emergency_vehicle = v_entry.vehicle_type == "emergency_vehicle"
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
        if vehicle_data_entry.vehicle_type == "emergency_vehicle":
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
        # parts.append(f"Relative Position: {f(vehicle_data_entry.relative_position, precision)}")
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

    @classmethod
    def summarize(
        cls,
        vehicle_data: VehicleData,
    ) -> str:
        if vehicle_data is None:
            return ""

        def lane_phrase(lane_name: str) -> str:
            if lane_name == "ego":
                return "in the ego lane"
            return f"in the {lane_name} lane"

        def crossing_phrase(v: VehicleDataEntry) -> str:
            y_pos = v.relative_position[1]
            theta = v.relative_orientation

            is_left_side = y_pos < 0
            is_moving_right = math.sin(theta) > 0

            side_str = "left side" if is_left_side else "right side"
            dir_str = "left to right" if is_moving_right else "right to left"
            is_towards = (is_left_side == is_moving_right)

            if is_towards:
                return f"and is moving {dir_str} from the {side_str} toward the ego path"
            return f"and is moving {dir_str} from the {side_str} away from the ego path"

        def build_sentences(
            traffic_category: str,
            group: Dict[str, List[VehicleDataEntry]]
        ) -> List[str]:
            sentences: List[str] = []
            if not group:
                return sentences

            for lane_name, entries in group.items():
                for v in entries:
                    is_cyclist = v.vehicle_type == "cyclist"
                    is_emergency = v.vehicle_type == "emergency_vehicle"
                    sirens_active = getattr(v, "emergency_sirens_active", False)

                    if traffic_category == "trailing" and lane_name == "ego" and not is_cyclist and not is_emergency:
                        continue

                    noun = "vehicle"
                    if is_cyclist:
                        noun = "cyclist"
                    elif is_emergency and sirens_active:
                        noun = "emergency vehicle"

                    intrudes_ego = False
                    if getattr(v, "intrudes_ego", False):
                        if getattr(v, "traffic_type", "") == "oncoming":
                            intrudes_ego = True
                        elif getattr(v, "traffic_type", "") == "leading" and is_cyclist:
                            intrudes_ego = True

                    caution = is_cyclist or (is_emergency and sirens_active) or intrudes_ego
                    prefix = "[CAUTION] " if caution else ""

                    dist = round(v.relative_distance)
                    sentence = (
                        f"{prefix}{traffic_category.capitalize()} {noun} {lane_phrase(lane_name)} "
                        f"is {dist} m away"
                    )

                    if v.speed >= 0.5:
                        sentence += f" moving at {round(v.speed)} m/s"

                    if traffic_category == "crossing":
                        sentence += f" {crossing_phrase(v)}"

                    if intrudes_ego:
                        sentence += " and is intruding into the ego lane"

                    sentence += "."
                    sentences.append(sentence)

            return sentences

        leading = build_sentences("leading", vehicle_data.view.get("leading", {}))
        oncoming = build_sentences("oncoming", vehicle_data.view.get("oncoming", {}))
        crossing = build_sentences("crossing", vehicle_data.view.get("crossing", {}))
        trailing = build_sentences("trailing", vehicle_data.view.get("trailing", {}))

        all_threats = leading + oncoming + crossing + trailing

        if not all_threats:
            return ""

        return "Dynamic Actor Context:\n" + "\n".join(f"- {s}" for s in all_threats)