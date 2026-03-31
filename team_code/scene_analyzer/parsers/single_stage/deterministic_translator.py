"""
Deterministic Policy Translator
================================
Converts an NSHighLevelBehaviour (Layer A output) into an EgoPlan (Layer B output)
without any VLM call. All logic is rule-based.
"""
import itertools
from typing import List

from team_code.scene_analyzer.parsers.single_stage.hl_beh_pydantic_models import HighLevelBehaviour, SemanticZone
from team_code.scene_analyzer.parsers.ego_plan_pydantic_models import (
    EgoPlan, ConditionCommand, ConditionTarget,
)

# ── Public API ────────────────────────────────────────────────────────────────

def translate(hl_beh: HighLevelBehaviour) -> EgoPlan:
    """
    Deterministically convert a HighLevelBehaviour (NS schema) into an EgoPlan.

    For every conflict zone, a ConditionCommand is generated for EVERY permutation
    of actor_type and traffic_type using itertools.product.
    """
    conditions: List[ConditionCommand] = []

    zones = hl_beh.active_conflict_zones + hl_beh.monitor_zones
    for zone in zones:
        # Note: 'none' is not in the Pydantic Literal, but kept defensively
        # in case of unexpected VLM string generation.
        if getattr(zone, "risk_level", "") == "none":
            continue

        # Unpack lists and create a condition target for each permutation
        for actor_type, traffic_type in itertools.product(zone.entity_types, zone.traffic_types):
            target = ConditionTarget(
                actor_type=actor_type,
                traffic_type=traffic_type,
                region=zone.region,
                description=zone.rationale,
            )

            conditions.append(
                ConditionCommand(
                    condition_action=zone.zone_constraint,
                    target=target,
                    priority=zone.risk_level,
                )
            )

    # target_speed: use speed limit when positive, else None (stop)
    target_speed = None

    # Reasoning: tag as deterministic + carry through VLM reasoning (up to 4 entries)
    reasoning = hl_beh.reasoning

    return EgoPlan(
        action=hl_beh.chosen_intent,
        target_speed=target_speed,
        conditions=conditions,
        reasoning=reasoning,
    )