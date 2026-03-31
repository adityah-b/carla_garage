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

    @classmethod
    def summarize(
        cls,
        traffic_data: TrafficData,
    ) -> str:
        if traffic_data is None:
            return ""

        bullets = []

        tl = traffic_data.next_traffic_light
        ss = traffic_data.next_stop_sign

        if tl:
            tl_dist = round(tl.distance_to_light)
            if tl.state in ("RED", "YELLOW"):
                bullets.append(
                    f"[CAUTION] Traffic light is {tl.state} and {tl_dist} m ahead."
                )
            else:
                bullets.append(
                    f"Traffic light is {tl.state} and {tl_dist} m ahead."
                )

        if ss:
            ss_dist = round(ss.distance_to_stop_sign)
            if ss.cleared:
                bullets.append(
                    f"Stop sign is {ss_dist} m ahead and has already been cleared."
                )
            else:
                bullets.append(
                    f"[CAUTION] Stop sign is {ss_dist} m ahead and has not been cleared yet."
                )

        bullets.append(f"Speed limit is {round(traffic_data.speed_limit)} m/s.")

        return "Traffic Context:\n" + "\n".join(f"- {b}" for b in bullets)