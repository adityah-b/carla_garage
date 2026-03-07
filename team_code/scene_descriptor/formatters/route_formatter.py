from typing import Dict, List

from agents.navigation.local_planner import RoadOption
from scene_descriptor.data_extractors.route_extractor import (
    RouteData,
    LaneChangeData,
    IntersectionData,
    IntersectionType,
    LaneInfo,
)

from .base_formatter import BaseFormatter

class RouteFormatter(BaseFormatter):
    @classmethod
    def format_route(
        cls,
        route_data : RouteData,
        precision : int = 2
    ) -> str:
        f = cls.fmt
        lines : List[str] = ["Route Data:"]

        lane_info = route_data.lane_info
        lc_data = route_data.lane_change_data
        i_data = route_data.intersection_data

        if lane_info:
            lines.append("\tLanes:")
            lines.append(cls._format_lane_info(lane_info=lane_info, indent="\t\t"))

        if lc_data:
            lines.append("\tLane Change:")
            lines.append(cls._format_lane_change_data(lc_data=lc_data, indent="\t\t", precision=precision))

        if i_data:
            lines.append("\tIntersection:")
            lines.append(cls._format_intersection_data(i_data=i_data, indent="\t\t", precision=precision))

        if len(lines) <= 1:
            lines = []

        return "\n".join(lines)

    @classmethod
    def _format_lane_info(
        cls,
        lane_info: LaneInfo,
        indent: str = ""
    ) -> str:
        def side_sentence(
            name: str,
            has_lane: bool,
            is_oncoming: bool,
            can_change: bool,
            same_dir_changes_available: bool,
        ) -> str:
            # No adjacent lane on that side
            if not has_lane:
                return (
                    f"{name} lane is not available because there is no "
                    f"adjacent lane on the {name.lower()} side."
                )

            # Oncoming lane (used for overtaking, must yield)
            if is_oncoming:
                return (
                    f"{name} lane is an oncoming lane that may be used briefly "
                    f"to overtake obstacles if necessary, but the ego vehicle must always yield "
                    f"to oncoming traffic."
                )

            # Same-direction lane
            if can_change:
                return (
                    f"{name} lane in same-direction traffic is available."
                )
            else:
                return (
                    f"{name} lane in same-direction traffic is available but not permitted."
                )

        left_sentence = side_sentence(
            "LEFT",
            lane_info.has_left_lane,
            lane_info.left_oncoming,
            lane_info.can_change_left,
            lane_info.same_direction_lane_change_available,
        )

        right_sentence = side_sentence(
            "RIGHT",
            lane_info.has_right_lane,
            lane_info.right_oncoming,
            lane_info.can_change_right,
            lane_info.same_direction_lane_change_available,
        )

        return f"{indent}{left_sentence}\n{indent}{right_sentence}"

    @classmethod
    def _format_lane_change_data(
        cls,
        lc_data : LaneChangeData,
        indent : str = "",
        precision : int = 2
    ) -> str:
        f = cls.fmt

        direction = "LEFT" if lc_data.target_maneuver == RoadOption.CHANGELANELEFT else "RIGHT"

        if not lc_data.inside_lane_change:
            return f"{indent}Change lane to the {direction} after driving {f(lc_data.distance_to_lane_change, precision)} metres"

        return f"{indent}Changing lane to the {direction}"

    @classmethod
    def _format_intersection_data(
        cls,
        i_data : IntersectionData,
        indent : str = "",
        precision : int = 2
    ) -> str:
        f = cls.fmt

        if i_data.target_maneuver == RoadOption.LEFT:
            turn = "Turn LEFT"
        elif i_data.target_maneuver == RoadOption.RIGHT:
            turn = "Turn RIGHT"
        else:
            turn = "Turn STRAIGHT"

        if i_data.signalized == IntersectionType.SIGNALIZED:
            i_type = "SIGNALIZED"
        elif i_data.signalized == IntersectionType.UNSIGNALIZED:
            i_type = "UNSIGNALIZED"
        else:
            i_type = "JUNCTION"

        if i_data.inside_intersection:
            return f"{indent}Driving through {i_type} intersection. Executing {turn} maneuver"

        if i_data.distance_to_intersection < 10.0:
            # return f"{indent}{turn} after arriving at {i_type} intersection in next {f(i_data.distance_to_intersection, precision)} metres"
            return f"{indent}{turn} at {i_type} intersection"

        return f"{indent}Approaching {i_type} intersection"

        # if not i_data.is_executing_maneuver:
        #     return f"{indent}{turn} at {i_type} intersection"

        # return f"{indent}Executing {turn} maneuver at {i_type} intersection"