from typing import Dict, List

from agents.navigation.local_planner import RoadOption
from scene_descriptor.data_extractors.route_extractor import (
    RouteData,
    LaneChangeData,
    IntersectionData,
    LaneInfo,
)
from privileged_route_planner import IntersectionType

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
            same_dir: bool,
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
            if same_dir:
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
            lane_info.left_same_dir,
            lane_info.same_direction_lane_change_available,
        )

        right_sentence = side_sentence(
            "RIGHT",
            lane_info.has_right_lane,
            lane_info.right_oncoming,
            lane_info.right_same_dir,
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
            # return f"{indent}Change lane to the {direction} after driving {f(lc_data.distance_to_lane_change, precision)} metres"
            return f"{indent}Change lane to the {direction}"

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

    @classmethod
    def summarize(
        cls,
        route_data: RouteData,
        precision: int = 2
    ) -> str:
        if route_data is None:
            return ""

        route_bullets = []
        lane_bullets = []

        lc = route_data.lane_change_data
        i = route_data.intersection_data
        li = route_data.lane_info

        if i:
            turn_map = {
                RoadOption.LEFT: "turn left",
                RoadOption.RIGHT: "turn right",
            }
            turn = turn_map.get(i.target_maneuver, "continue straight")

            i_type = {
                IntersectionType.SIGNALIZED: "signalized intersection",
                IntersectionType.UNSIGNALIZED: "unsignalized intersection",
            }.get(i.signalized, "junction")

            if i.inside_intersection:
                route_bullets.append(
                    f"The ego is inside a {i_type} and is executing a maneuver to {turn}."
                )
            elif i.distance_to_intersection < 10.0:
                route_bullets.append(
                    f"The ego is at a {i_type} and should {turn}."
                )
            else:
                route_bullets.append(
                    f"The route requires the ego to {turn} at an upcoming {i_type}."
                )

        elif lc:
            direction = "left" if lc.target_maneuver == RoadOption.CHANGELANELEFT else "right"
            if lc.inside_lane_change:
                route_bullets.append(
                    f"[CAUTION] The ego is currently changing lanes to the {direction}."
                )
            else:
                route_bullets.append(
                    f"[CAUTION] The route requires a lane change to the {direction}."
                )

        else:
            route_bullets.append("The ego should continue following the current route.")

        if li:
            if li.has_left_lane:
                if li.left_oncoming:
                    lane_bullets.append("The left adjacent lane is oncoming but can be used briefly for overtakes.")
                elif li.left_same_dir:
                    lane_bullets.append("A same-direction lane is available on the left.")
                else:
                    lane_bullets.append("A left adjacent lane exists but is not permitted for same-direction travel.")
            else:
                lane_bullets.append("There is no adjacent lane on the left.")

            if li.has_right_lane:
                if li.right_oncoming:
                    lane_bullets.append("The right adjacent lane is oncoming but can be used briefly for overtakes.")
                elif li.right_same_dir:
                    lane_bullets.append("A same-direction lane is available on the right.")
                else:
                    lane_bullets.append("A right adjacent lane exists but is not permitted for same-direction travel.")
            else:
                lane_bullets.append("There is no adjacent lane on the right.")

        sections = []

        if route_bullets:
            sections.append("Route Navigation Context:\n" + "\n".join(f"- {b}" for b in route_bullets))

        if lane_bullets:
            sections.append("Lane Topology Context:\n" + "\n".join(f"- {b}" for b in lane_bullets))

        return "\n\n".join(sections)