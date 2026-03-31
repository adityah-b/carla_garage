from pydantic import BaseModel, Field
from typing import Any, List, Literal, Optional, Deque, Union
from enum import Enum
from dataclasses import dataclass

from team_code.scene_analyzer.parsers.base_pydantic_models import *

class ConditionTarget(BaseModel):
    """Single-actor target that the low-level planner resolves against live perception each tick."""
    actor_type: EntityType
    traffic_type: TrafficType
    region: RegionType

class ConditionCommand(BaseModel):
    condition_action: ConditionAction
    target: ConditionTarget
    priority: Priority = "medium"

class EgoPlan(BaseModel):
    action: Action
    target_speed: Optional[float] = None
    conditions: List[ConditionCommand]
    reasoning: List[str] = Field(
        min_length=1,
        max_length=5,
        description="Step by step reasoning on why each step and parameter choice is valid",
    )

    def to_string(self) -> str:
        lines: List[str] = []

        lines.append("DRIVING ACTION:")
        lines.append(self.action.value)
        lines.append("")

        lines.append("CONDITIONS:")
        if self.conditions:
            for condition in self.conditions:
                t = condition.target
                cond_desc = (
                    f"- {condition.condition_action.value} "
                    f"{t.actor_type} [{t.region}] "
                    f"traffic={t.traffic_type} "
                    f"priority={condition.priority}"
                )
                lines.append(cond_desc)
        else:
            lines.append("- none")

        lines.append("")
        lines.append("REASONING:")
        for idx, reason in enumerate(self.reasoning, start=1):
            lines.append(f"{idx}. {reason}")

        return "\n".join(lines)


class PlanStatus(str, Enum):
    EXECUTING = "executing"
    FINISHED = "finished"
    FAILED = "failed"

@dataclass
class PlanState:
    """Bundles all state for a plan execution — current or archived."""
    plan: EgoPlan
    hl_beh: Any  # HighLevelBehaviour from either single_stage/ or dual_stage/ pipeline
    status: PlanStatus
    reason: Optional[str] = None
    collision_events: Optional[Deque] = None
    # Scene state captured at the moment the plan was created — used for memory logging
    scene_text: Optional[str] = None
    scene_image: Any = None  # np.ndarray | None

    def to_string(self) -> str:
        lines = []
        lines.append(f"action: {self.plan.action.value}")
        lines.append("conditions:")
        if self.plan.conditions:
            for condition in self.plan.conditions:
                t = condition.target
                lines.append(
                    f"- {condition.condition_action.value} "
                    f"{t.actor_type} [{t.region}] "
                    f"traffic={t.traffic_type}"
                )
        else:
            lines.append("- none")
        status_line = f"status: {self.status.value}"
        if self.reason:
            status_line += f" (reason: {self.reason})"
        lines.append(status_line)
        if self.collision_events:
            lines.append("HAS COLLISIONS")
            for event in self.collision_events:
                lines.append(f"- Actor ID: {event.id}, time: {event.timestamp}")
        return "\n".join(lines)
