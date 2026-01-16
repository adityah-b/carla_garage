from typing import Dict, List
from .base_formatter import BaseFormatter
from scene_descriptor.data_extractors.traffic_data_extractor import TrafficData, TrafficLightData, StopSignData

class TrafficFormatter(BaseFormatter):
    @classmethod
    def format_traffic(
        cls,
        traffic_data: TrafficData,
        precision: int = 2
    ) -> str:
        f = cls.fmt
        lines : List[str] = ["Traffic Data:"]

        tl_data = traffic_data.next_traffic_light
        ss_data = traffic_data.next_stop_sign
        speed_limit = traffic_data.speed_limit

        if tl_data:
            lines.append("\tNext Traffic Light:")
            lines.append(cls._format_traffic_light_data(tl_data=tl_data, indent="\t\t", precision=precision))

        if ss_data:
            lines.append("\tNext Stop Sign:")
            lines.append(cls._format_stop_sign_data(ss_data=ss_data, indent="\t\t", precision=precision))

        lines.append(f'\tSpeed Limit: {f(speed_limit, precision)}')

        return '\n'.join(lines)

    @classmethod
    def _format_traffic_light_data(
        cls,
        tl_data : TrafficLightData,
        indent : str = "",
        precision : int = 2
    ) -> str:
        f = cls.fmt
        return (
            f"{indent}Traffic Light ID: {tl_data.id}, "
            f"State: {tl_data.state}, "
            f"Relative Distance: {f(tl_data.distance_to_light, precision)}"
        )

    @classmethod
    def _format_stop_sign_data(
        cls,
        ss_data : StopSignData,
        indent : str = "",
        precision : int = 2
    ) -> str:
        f = cls.fmt
        # TODO: UPDATE THE FORMATTING, MAKE IT MORE EXPRESSIVE
        return (
            f"{indent}Stop Sign ID: {ss_data.id}, "
            f"Relative Distance: {f(ss_data.distance_to_stop_sign, precision)}, "
            f"Cleared: {ss_data.cleared}"
        )