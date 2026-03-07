from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Set, Tuple

from team_code.scene_analyzer.parsers.ego_plan_pydantic_models import ConditionCommand, Action

class ActionPhase(Enum):
    IDLE = "idle"
    EXECUTING = "executing"
    CLEARED = "cleared"


class ConditionStatus:
    def __init__(
        self,
        condition: ConditionCommand,
        phase: ActionPhase,
        reason: str,
        is_blocking: bool = False,
        needs_replan : bool = False,
    ) -> None:
        self.condition = condition
        self.phase = phase
        self.reason = reason
        self.is_blocking = is_blocking
        self.needs_replan = needs_replan

class CommandStatus:
    """
    Structured command output fed back to VLM
    """

    def __init__(
        self,
        cur_cmd: Action,
        phase: ActionPhase,
        is_blocked: bool = False,
        conditions_status: Optional[List[ConditionStatus]] = None,
        reasons: Optional[List[str]] = None,
    ) -> None:
        self.cur_cmd = cur_cmd
        self.phase = phase
        self.is_blocked = is_blocked
        # avoid mutable default pitfall
        self.conditions_status = conditions_status if conditions_status is not None else []
        self.reasons = reasons if reasons is not None else []

    def to_string(self) -> str:
        lines: List[str] = []
        lines.append(f"action: {self.cur_cmd.value}")
        lines.append(f"phase: {self.phase.value}")
        lines.append(f"is_blocked: {self.is_blocked}")

        lines.append("conditions:")
        if self.conditions_status:
            for cond_status in self.conditions_status:
                cond_desc = (
                    f"- {cond_status.condition.condition_action.value} "
                    f"{cond_status.condition.traffic_type} "
                    f"{cond_status.condition.obj_type} "
                    f"{cond_status.condition.id}"
                    f"\n\t- phase: {cond_status.phase.value}"
                    f"\n\t- is_blocking: {cond_status.is_blocking}"
                )
                lines.append(cond_desc)
        else:
            lines.append("- none")

        lines.append("reasons:")
        if self.reasons:
            for reason in self.reasons:
                lines.append(f"- {reason}")
        else:
            lines.append("- none")

        return "\n".join(lines)