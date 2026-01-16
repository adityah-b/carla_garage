# from pydantic import BaseModel, Field
# from typing import List, Literal

# class KeyActor(BaseModel):
#     id : int
#     obj_type : Literal["vehicle", "ped", "cyclist", "traffic_light", "stop_sign", "obstacle"]
#     traffic_type : Literal["leading", "trailing", "oncoming", "cross", "other"]

# class HighLevelBehaviour(BaseModel):
#     scenario : str = Field(description="one paragraph natural language summary describing the driving scenario from the provided image and text along with the necessary high-level driving behaviour to navigate the scenario")
#     key_actors : List[KeyActor]
#     reasoning : List[str] = Field(min_length=1, description="Step by step reasoning on why each action and key actor choice is valid")

from pydantic import BaseModel, Field
from typing import List, Literal


InfluenceOnEgo = Literal[
    "NO_INFLUENCE",
    "AFFECTS_SPEED",
    "AFFECTS_ROUTE",
    "AFFECTS_SPEED_AND_ROUTE",
]

class RouteGuidance(BaseModel):
    maneuver: Literal[
        "GO_STRAIGHT",
        "TURN_LEFT",
        "TURN_RIGHT",
    ] = "FOLLOW_ROUTE"
    intersection_type: Literal["signalized", "unsignalized", "none"] = "none"
    distance_to_route_point_m: float = 0.0


class KeyActor(BaseModel):
    # ([traffic_type]) [actor_type] [id] is [distance_from_ego_m] m away (going [speed_mps] m/s)
    id: int
    actor_type: Literal["vehicle", "pedestrian", "cyclist", "emergency vehicle"]
    traffic_type: Literal["leading", "trailing", "oncoming", "cross", "other"]
    traffic_lane: Literal["ego lane", "right lane", "left lane", "other"]
    distance_from_ego_m: float
    speed_mps: float

class TrafficObject(BaseModel):
    # [object_type] [id] is [distance_to_object_m] m ahead (with state [state])
    id: int
    object_type: Literal["traffic_light", "stop_sign"]
    distance_to_object_m: float
    state: Literal["GREEN", "RED", "STOP_SIGN"]

class Obstacle(BaseModel):
    # [obstacle_type] is [distance_to_object_m] m ahead
    id: int
    obstacle_type: Literal[
        "road_hazard",
        "construction",
        "stopped_vehicle",
        "parked_vehicle",
        "debris",
        "other",
    ]
    distance_to_object_m: float

class HighLevelBehaviour(BaseModel):
    # current_action: Literal[
    #     "FOLLOW_ROUTE",
    #     "STOPPED",
    #     "GO_STRAIGHT",
    #     "TURN_LEFT",
    #     "TURN_RIGHT",
    #     "CHANGE_LANE_LEFT",
    #     "CHANGE_LANE_RIGHT",
    #     "OVERTAKE",
    #     "EMERGENCY_STOP",
    # ] = "FOLLOW_ROUTE"

    # route_guidance: RouteGuidance
    key_actors: List[KeyActor]
    traffic_objects: List[TrafficObject]
    obstacles: List[Obstacle]
    next_action: str
    reasoning: List[str] = Field(
        min_length=1,
        max_length=10,
        description="Step-by-step reasoning supporting the influences and next_action.",
    )

    def to_string(self) -> str:
        lines: List[str] = []

        lines.append("KEY ACTORS:")
        if not self.key_actors:
            lines.append("None.")
        else:
            for a in self.key_actors:
                traffic_lane_info = ""
                if a.traffic_lane != 'other':
                    traffic_lane_info = f' in {a.traffic_lane}'

                base = (
                    f"- {a.traffic_type} {a.actor_type} {a.id} {traffic_lane_info} is "
                    f"{a.distance_from_ego_m:.2f} meters away"
                )

                speed_part = ""
                # Only show speed where it makes sense and is non-trivial
                if a.speed_mps > 0.05 and a.actor_type in [
                    "vehicle",
                    "cyclist",
                    "emergency",
                ]:
                    speed_part = f" going {a.speed_mps:.2f} m/s"

                lines.append(base + speed_part)
        lines.append("")

        # === TRAFFIC OBJECTS ===
        # template: [object_type] [id] is [distance] ahead (with state [state])
        lines.append("TRAFFIC OBJECTS:")
        if not self.traffic_objects:
            lines.append("None.")
        else:
            for t in self.traffic_objects:
                base = (
                    f"- {t.object_type} {t.id} is "
                    f"{t.distance_to_object_m:.2f} meters ahead"
                )

                state_part = ""
                # Only lights have meaningful states
                if t.object_type == "traffic_light":
                    state_part = f" (with state {t.state})"

                lines.append(base + state_part)
        lines.append("")

        # === OBSTACLES ===
        # template: [obstacle_type] is [distance] ahead
        lines.append("OBSTACLES:")
        if not self.obstacles:
            lines.append("None.")
        else:
            for o in self.obstacles:
                base = (
                    f"- {o.obstacle_type} {o.id} is "
                    f"{o.distance_to_object_m:.2f} meters ahead"
                )
                lines.append(base)
        lines.append("")

        # === NEXT ACTION ===
        lines.append("NEXT ACTION:")
        lines.append(self.next_action)
        lines.append("")

        # === REASONING ===
        lines.append("REASONING:")
        for idx, reason in enumerate(self.reasoning, start=1):
            lines.append(f"{idx}. {reason}")

        return "\n".join(lines)


