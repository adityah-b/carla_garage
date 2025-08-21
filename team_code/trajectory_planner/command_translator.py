import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Union, Optional


class Action(str, Enum):
    ACCELERATE = "accelerate"
    DECELERATE = "decelerate"
    MAINTAIN_SPEED = "maintain_speed"
    CHANGE_LANE_LEFT = "change_lane_left"
    CHANGE_LANE_RIGHT = "change_lane_right"
    STOP = "stop"

    @staticmethod
    def from_token(token: str) -> "Action":
        t = token.strip().lower().replace("-", "_")
        t = re.sub(r"\s+", "_", t)
        try:
            return Action(t)
        except ValueError as e:
            raise ValueError(f"Unknown action: '{token}'") from e


@dataclass(frozen=True)
class EgoPlan:
    raw_str: str
    plan: List[str]            # e.g., ["accelerate", "maintain_speed", ...]
    low_level_actions: List[Dict]  # each has "longitudinal_params"/"lateral_params"
    reasoning: str


@dataclass(frozen=True)
class Params:
    spd: Optional[Union[float, str]]
    t_head: Optional[float]
    f_dist: Optional[float]
    gap_time: Optional[float]
    gap_dist: Optional[float]


@dataclass(frozen=True)
class Plan:
    actions: List[Action]   # High-level action sequence
    params: List[Params]    # Detailed actions with parameters


class CommandTranslator:
    """
    Stateless translator that converts an EgoPlan object to a normalized Plan object.
    """

    @staticmethod
    def translate_plan(ego_plan: EgoPlan) -> Plan:
        """Convert EgoPlan to normalized Plan."""

        # Normalize actions -> Action enums
        actions: List[Action] = []
        for action_str in ego_plan.plan:
            try:
                actions.append(Action.from_token(action_str))
            except ValueError as e:
                raise ValueError(f"Invalid action in plan: '{action_str}'") from e

        # Map parameters:
        # - longitudinal: spd, t_head, f_dist
        # - lateral: gap_time, gap_dist
        params: List[Params] = []
        for action_dict in ego_plan.low_level_actions:
            longitudinal = action_dict.get("longitudinal_params", {}) or {}
            lateral = action_dict.get("lateral_params", {}) or {}

            p = Params(
                spd=longitudinal.get("spd"),
                t_head=CommandTranslator._to_float_or_none(longitudinal.get("t_head")),
                f_dist=CommandTranslator._to_float_or_none(longitudinal.get("f_dist")),
                gap_time=CommandTranslator._to_float_or_none(lateral.get("gap_time")),
                gap_dist=CommandTranslator._to_float_or_none(lateral.get("gap_dist")),
            )
            params.append(p)

        return Plan(actions=actions, params=params)

    @staticmethod
    def _to_float_or_none(v) -> Optional[float]:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None