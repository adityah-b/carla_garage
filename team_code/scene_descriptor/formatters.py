from typing import Dict, Any, List

class SceneFormatter:
    """
    Formats structured scene data into human-readable text representation.

    Responsibilities:
    - Convert structured data objects to formatted strings
    - Handle missing or null data gracefully
    - Provide consistent formatting across data types
    """

    def __init__(self, config=None):
        """Initialize formatter with optional configuration."""
        self.config = config

    def format_scene_data(self, structured_data: Dict[str, Any]) -> str:
        """
        Convert complete structured scene data to formatted string.

        Args:
            structured_data: Dictionary containing traffic, ego, and agent data

        Returns:
            Formatted string representation of the scene
        """
        sections = [
            self._format_traffic_data(structured_data.get('traffic')),
            self._format_ego_data(structured_data.get('ego')),
            self._format_agent_data(structured_data.get('agent'))
        ]

        return '\n'.join(filter(None, sections))

    def _format_traffic_data(self, traffic_data: Dict[str, Any]) -> str:
        """Format traffic control information."""
        if not traffic_data:
            return ""

        lines = ["Traffic Data:"]

        # Traffic light information
        lines.append("    Next Traffic Light:")
        traffic_light = traffic_data.get('next_traffic_light')
        if traffic_light:
            lines.append(
                f"        Traffic Light ID: {traffic_light.get('id', 'N/A')}, "
                f"State: {traffic_light.get('state', 'N/A')}, "
                f"Relative Distance: {traffic_light.get('distance_to_light', 'N/A')}"
            )
        else:
            lines.append("        No data available")

        # Stop sign information
        lines.append("    Next Stop Sign:")
        stop_sign = traffic_data.get('next_stop_sign')
        if stop_sign:
            lines.append(
                f"        Distance to Stop Sign: {stop_sign.get('distance_to_stop_sign', 'N/A')}"
            )
        else:
            lines.append("        No data available")

        # Speed limit
        speed_limit = traffic_data.get('speed_limit', 'N/A')
        lines.append(f"    Speed Limit: {speed_limit}")

        return '\n'.join(lines)

    def _format_ego_data(self, ego_data: Dict[str, Any]) -> str:
        """Format ego vehicle information."""
        if not ego_data:
            return ""

        lines = ["Ego Data:"]
        lines.append(f"    Speed: {ego_data.get('speed', 'N/A')}")
        lines.append(f"    Orientation: {ego_data.get('orientation', 'N/A')}")
        lines.append(f"    Position: {ego_data.get('position', 'N/A')}")

        # Lane change information
        lines.append("    Lane Change:")
        lane_change = ego_data.get('lane_change', {})
        lines.extend([
            f"        Has Upcoming Lane Change: {lane_change.get('has_upcoming_lane_change', 'N/A')}",
            f"        Lane Change Direction: {lane_change.get('lane_change_direction', 'N/A')}",
            f"        Can Change Lane: {lane_change.get('can_change_lane', 'N/A')}",
            f"        Available Lane Change Distance: {lane_change.get('available_lane_change_distance', 'N/A')}"
        ])

        return '\n'.join(lines)

    def _format_agent_data(self, agent_data: Dict[str, Any]) -> str:
        """Format agent/vehicle information."""
        if not agent_data:
            return ""

        lines = ["Agent Data:"]

        # Format ongoing traffic
        ongoing_traffic = agent_data.get("Ongoing Traffic", {})
        if ongoing_traffic:
            lines.append("    Ongoing Traffic:")
            lines.extend(self._format_traffic_direction(ongoing_traffic, "        "))

        # Format oncoming traffic (commented out in original, but structure provided)
        oncoming_traffic = agent_data.get("Oncoming Traffic", {})
        if oncoming_traffic:
            lines.append("    Oncoming Traffic:")
            lines.extend(self._format_traffic_direction(oncoming_traffic, "        "))

        # Format cross traffic
        cross_traffic = agent_data.get("Cross Traffic", {})
        if cross_traffic:
            lines.append("    Cross Traffic:")
            lines.extend(self._format_traffic_direction(cross_traffic, "        "))

        return '\n'.join(lines)

    def _format_traffic_direction(self, traffic_direction: Dict[str, Any], indent: str) -> List[str]:
        """
        Format vehicles in a specific traffic direction.

        Args:
            traffic_direction: Dictionary of lanes with vehicle data
            indent: String indentation for formatting

        Returns:
            List of formatted lines
        """
        lines = []

        for lane_name, lane_data in traffic_direction.items():
            if not lane_data.get("leading_vehicles") and not lane_data.get("trailing_vehicles"):
                continue

            lines.append(f"{indent}{lane_name}:")

            # Leading vehicles
            lines.append(f"{indent}    Leading Vehicles:")
            leading_vehicles = lane_data.get("leading_vehicles", [])
            if leading_vehicles:
                for vehicle in leading_vehicles:
                    lines.append(self._format_vehicle_info(vehicle, f"{indent}        "))
            else:
                lines.append(f"{indent}        No data available")

            # Trailing vehicles
            lines.append(f"{indent}    Trailing Vehicles:")
            trailing_vehicles = lane_data.get("trailing_vehicles", [])
            if trailing_vehicles:
                for vehicle in trailing_vehicles:
                    lines.append(self._format_vehicle_info(vehicle, f"{indent}        "))
            else:
                lines.append(f"{indent}        No data available")

        return lines

    def _format_vehicle_info(self, vehicle: Dict[str, Any], indent: str) -> str:
        """
        Format individual vehicle information.

        Args:
            vehicle: Dictionary containing vehicle data
            indent: String indentation for formatting

        Returns:
            Formatted vehicle information string
        """
        vehicle_id = vehicle.get('vehicle_id', 'N/A')
        data = vehicle.get('data', {})

        return (
            f"{indent}Vehicle ID: {vehicle_id}, "
            f"Relative Position: {data.get('relative position', 'N/A')}, "
            f"Relative Orientation: {data.get('relative orientation', 'N/A')}, "
            f"Speed: {data.get('speed', 'N/A')}, "
            f"Relative Distance: {data.get('relative distance', 'N/A')}"
        )


class CompactSceneFormatter(SceneFormatter):
    """
    Compact version of scene formatter with reduced verbosity.

    Provides more concise output suitable for logging or compact displays.
    """

    def format_scene_data(self, structured_data: Dict[str, Any]) -> str:
        """Generate compact formatted representation."""
        sections = []

        # Compact traffic info
        traffic = structured_data.get('traffic')
        if traffic:
            traffic_info = []
            if traffic.get('next_traffic_light'):
                tl = traffic['next_traffic_light']
                traffic_info.append(f"TL:{tl.get('state', 'N/A')}@{tl.get('distance_to_light', 'N/A')}m")
            if traffic.get('next_stop_sign'):
                ss = traffic['next_stop_sign']
                traffic_info.append(f"SS@{ss.get('distance_to_stop_sign', 'N/A')}m")
            if traffic.get('speed_limit'):
                traffic_info.append(f"Limit:{traffic['speed_limit']}")

            if traffic_info:
                sections.append("Traffic: " + ", ".join(traffic_info))

        # Compact ego info
        ego = structured_data.get('ego')
        if ego:
            ego_info = [
                f"Speed:{ego.get('speed', 'N/A')}",
                f"Pos:{ego.get('position', 'N/A')}"
            ]
            lane_change = ego.get('lane_change', {})
            if lane_change.get('has_upcoming_lane_change'):
                ego_info.append(f"LC:{lane_change.get('lane_change_direction', 'N/A')}")

            sections.append("Ego: " + ", ".join(ego_info))

        # Compact agent info
        agent = structured_data.get('agent')
        if agent and agent.get("Ongoing Traffic"):
            vehicle_count = sum(
                len(lane_data.get("leading_vehicles", [])) + len(lane_data.get("trailing_vehicles", []))
                for lane_data in agent["Ongoing Traffic"].values()
            )
            if vehicle_count > 0:
                sections.append(f"Vehicles: {vehicle_count} nearby")

        return " | ".join(sections)
