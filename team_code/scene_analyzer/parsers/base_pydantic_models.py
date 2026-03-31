from pydantic import BaseModel, Field
from typing import List, Literal, Union
from enum import Enum

class EntityType(str, Enum):
    VEHICLE           = "vehicle"
    EMERGENCY_VEHICLE = "emergency_vehicle"
    CYCLIST           = "cyclist"
    PEDESTRIAN        = "pedestrian"
    STOP_LIGHT        = "stop_light"
    STOP_SIGN         = "stop_sign"
    OBSTACLE          = "obstacle"
    ANY               = "any"

TrafficType = Literal[
    "leading",
    "trailing",
    "oncoming",
    "crossing",
    "intruding",
]

RegionType = Literal[
    "ego_path",
    "adjacent_left",
    "adjacent_right",
    "blind_spot",
    "intersection",
]

Priority = Literal[
    "low",
    "medium",
    "high",
    "critical"
]

class Action(str, Enum):
    """Unified action enum shared across both planning stages for Two-Level Filtering."""

    # Longitudinal
    FOLLOW_ROUTE = "follow_route"

    # Turns
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    TURN_STRAIGHT = "turn_straight"

    # Lane changes
    CHANGE_LANE_LEFT = "change_lane_left"
    CHANGE_LANE_RIGHT = "change_lane_right"

    # Overtakes
    OVERTAKE_LEFT = "overtake_left"
    OVERTAKE_RIGHT = "overtake_right"

    # Pull over for emergency vehicles
    PULL_OVER_LEFT = "pull_over_left"
    PULL_OVER_RIGHT = "pull_over_right"
    PULL_OVER_IN_LANE = "pull_over_in_lane"

    # Lane share
    SHARE_LANE = "share_lane"

class ConditionAction(str, Enum):
    WATCH_FOR = "watch_for"
    YIELD_FOR = "yield_for"
    STOP_FOR = "stop_for"
    KEEP_DISTANCE_FROM = "keep_distance_from"
    PASS_WITH_CLEARANCE = "pass_with_clearance"
    GIVE_WAY_TO = "give_way_to"

class MemoryReflection(BaseModel):
    retrieved_similarities: List[str] = Field(
        min_length=1,
        max_length=3,
        description="Identify the key kinematic and visual overlaps between the past memories and the live scene.",
    )
    critical_differences: List[str] = Field(
        min_length=1,
        max_length=3,
        description="Identify what is NOVEL or DIFFERENT in the current live scene compared to the past memories.",
    )
    extrapolated_risk: List[str] = Field(
        min_length=1,
        max_length=3,
        description="Based on the critical differences, explain how the risk assessment must change compared to past memories.",
    )
