import enum

from pydantic import BaseModel, Field
from typing import Optional

class ActorType(enum.Enum):
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    TRAFFIC_LIGHT = "traffic_light"
    STOP_SIGN = "stop_sign"

class KeyActor(BaseModel):
    actor_type: ActorType
    actor_id: int

class LongitudinalCommand(enum.Enum):
    """
    Enum for longitudinal driving commands.
    """
    # MAINTAIN_SPEED = "maintain_speed"
    ACCELERATE = "accelerate"
    DECELERATE = "decelerate"
    CHANGE_LANE_LEFT = "change_lane_left"
    CHANGE_LANE_RIGHT = "change_lane_right"
    # STOP = "stop"

# class LongitudinalCommand(enum.Enum):
#     """
#     Enum for longitudinal driving commands.
#     """
#     FOLLOW_LEAD_VEHICLE = "follow_lead_vehicle"
#     FOLLOW_SPEED_LIMIT = "follow_speed_limit"
#     # ACCELERATE = "accelerate"
#     # DECELERATE = "decelerate"
#     CHANGE_LANE_LEFT = "change_lane_left"
#     # STOP = "stop"

class LongitudinalCommandParams(BaseModel):
    """
    Parameters for high-level driving commands.
    """
    # desired_duration: float = Field(None, ge=0, description="The desired duration to execute the maneuver, in seconds.")
    desired_following_distance: Optional[float] = Field(..., ge=2.0, description="The desired following distance to the leading vehicle, in meters.")
    target_speed: Optional[float] = Field(None, ge=0, description="The target speed to accelerate or decelerate to, in m/s.")

class HighLevelCommand(BaseModel):
    """
    High-level driving command to be executed by the
    low-level controller.
    """
    command: LongitudinalCommand = Field(..., description="The high-level driving command to execute.")
    params: LongitudinalCommandParams = Field(..., description="Parameters for the high-level driving command.")
    key_actors: list[KeyActor] = Field(..., description="Key actors involved in the command.")
    reasoning: str

class SceneDescription(BaseModel):
    """
    Description of the scene, including key actors and their types.
    """
    road_description: str = Field(..., description="Description of the road network and its conditions.")
    traffic_description: str = Field(..., description="Description of the traffic conditions in the scene.")
    static_objects_and_obstacles_description: str = Field(..., description="Description of static objects and obstacles in the scene.")
    ego_vehicle_description: str = Field(..., description="Description of the ego vehicle and its state.")
    key_actors: list[KeyActor] = Field(..., description="List of key actors in the scene.")
    reasoning: str = Field(..., description="Reasoning for the chosen key actors")

class VehicleConditions(BaseModel):
    class VehicleTrafficType(enum.Enum):
        ONGOING = "ongoing"
        ONCOMING = "oncoming"
        CROSS = "cross"

    vehicle_id: int = Field(..., ge=0, description="The ID of the vehicle.")
    vehicle_traffic_type: VehicleTrafficType = Field(..., description="The type of traffic the vehicle is in.")

    ### Spatial conditions ###

    # Condition to check if the vehicle has passed the ego vehicle.
    # Examples:
    # Ongoing traffic:
    # - Vehicle has gone ahead of the ego vehicle
    # Oncoming traffic:
    # - Vehicle has gone past the ego vehicle in the opposite direction
    # Cross traffic:
    # - Vehicle has crossed the ego vehicle's path
    has_passed_ego: Optional[bool] = Field(None, description="Whether the vehicle has passed the ego vehicle based on its traffic type.")

    # Condition to check if sufficient gap is available for the vehicle to merge.
    # NOTE: Only applicable for ongoing traffic.
    has_sufficient_gap: Optional[bool] = Field(None, description="Whether the ego vehicle has sufficient gap to merge into the target lane near this vehicle (only applies to ongoing traffic).")

    # Condition to check if the vehicle has crossed an intersection. Useful in cases where the vehicle does not explicitly move past the ego vehicle.
    # Examples:
    # Ongoing/oncoming/cross traffic:
    # - Vehicle has crossed the intersection (either turned left or right)
    has_crossed_intersection: Optional[bool] = Field(None, description="Whether the vehicle has crossed an intersection.")

    # Debugging
    description: Optional[str] = Field(None, description="Human-readable description of the condition, for debugging or explanation.")

class EndConditions(BaseModel):
    vehicle_conditions: Optional[list[VehicleConditions]] = None