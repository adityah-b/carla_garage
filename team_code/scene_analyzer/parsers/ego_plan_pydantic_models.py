from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from enum import Enum

class Action(str, Enum):
    FOLLOW_ROUTE = "follow_route"
    STOP_FOR = "stop_for"
    YIELD_FOR = "yield_for"
    CHANGE_LANE_LEFT = "change_lane_left"
    CHANGE_LANE_RIGHT = "change_lane_right"

class LongitudinalParams(BaseModel):
    spd: Optional[float] = Field(default=None, ge=0.0, description="Desired absolute speed setpoint in m/s")
    t_head: Optional[float] = Field(default=1.5, ge=1.0, le=2.5, description="Desired time headway in seconds. Larger = more cautious")
    f_dist: Optional[float] = Field(default=4.0, ge=2.0, le=8.0, description="Desired following distance in metres. Larger = more cautious")

class LateralParams(BaseModel):
    gap_time: float = Field(default=1.5, ge=1.0, le=3.0, description="Minimum acceptable time gap in target lane (ahead and behind) in seconds")
    gap_dist: float = Field(default=10.0, ge=4.0, le=30.0, description="Minimum acceptable distance margins in target lane (ahead and behind) in metres")

class ConditionalParams(BaseModel):
    target : Literal["vehicle", "cyclist", "ped", "obstacle", "stop_sign", "traffic_light"] = Field(description="The type of the target object")
    id : int = Field(description="The actor ID of the chosen target")

class LowLevelAction(BaseModel):
    action : Action
    longitudinal_params : Optional[LongitudinalParams] = None
    lateral_params : Optional[LateralParams] = None
    conditional_params : Optional[ConditionalParams] = None

class EgoPlan(BaseModel):
    plan : List[LowLevelAction] = Field(min_length=1)
    reasoning : List[str] = Field(min_length=1, description="Step by step reasoning on why each step and parameter choice is valid")
