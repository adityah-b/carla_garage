from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Set, Tuple

from team_code.scene_analyzer.parsers.ego_plan_pydantic_models import ConditionCommand, Action

class ActionPhase(Enum):
    IDLE = "idle"
    EXECUTING = "executing"
    CLEARED = "cleared"
    FAILED = "failed"

class LateralPlanStatus(Enum):
    NONE = auto()
    VALID = auto()
    INVALID_TRANSIENT = auto()
    INVALID_FATAL = auto()

class OvertakeType(Enum):
    LANE_CHANGE = auto()  # overtake via an available parallel driving lane
    ONCOMING    = auto()  # overtake by invading the oncoming traffic lane

class OvertakeSubPhase(Enum):
    PENDING_PLAN   = auto()  # waiting for first valid LatPlanner result
    LC_OUTBOUND    = auto()  # transitioning to the target lane
    IN_TARGET_LANE = auto()  # in target lane, IDM-following until clear to return
    LC_RETURN      = auto()  # transitioning back to the source lane

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
                t = cond_status.condition.target
                cond_desc = (
                    f"- {cond_status.condition.condition_action.value} "
                    f"{t.actor_type} [{t.region}] "
                    f"traffic={t.traffic_type}"
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