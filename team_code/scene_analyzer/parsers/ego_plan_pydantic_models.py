from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Tuple
from enum import Enum

class Action(str, Enum):
    ########################################
    # LONGITUDINAL
    ########################################

    FOLLOW_ROUTE = "follow_route"
    # HOLD_POSITION = "hold_position"

    ########################################
    # LATERAL
    ########################################

    # Turns
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"

    # Lane changes
    CHANGE_LANE_LEFT = "change_lane_left"
    CHANGE_LANE_RIGHT = "change_lane_right"

    # Overtakes
    OVERTAKE_LEFT = "overtake_left"
    OVERTAKE_RIGHT = "overtake_right"

class ConditionAction(str, Enum):
    YIELD_FOR = "yield_for"
    STOP_FOR = "stop_for"
    WATCH_OUT_FOR = "watch_out_for"

class ConditionCommand(BaseModel):
    condition_action : ConditionAction
    id : int = Field(description="The actor ID of the chosen target")
    obj_type : Literal["vehicle", "cyclist", "ped", "obstacle", "stop_sign", "traffic_light"] = Field(description="The type of the target object")
    importance : float = 1.0

class EgoPlan(BaseModel):
    # High-level action
    action : Action

    # Optional adjustment parameters
    target_speed : Optional[float] = None
    # target_route_adjustments : Optional[List[Tuple[float, float]]] = None

    # Conditions
    conditions : List[ConditionCommand]
    reasoning : List[str] = Field(min_length=1, description="Step by step reasoning on why each step and parameter choice is valid")
