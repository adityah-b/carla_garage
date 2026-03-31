from typing import List
from pydantic import BaseModel, Field

from team_code.scene_analyzer.parsers.base_pydantic_models import *

class SemanticZone(BaseModel):
    """A semantic zone enriched with the action hint the Deterministic Translator should generate."""
    region: RegionType

    entity_types: List[EntityType]
    traffic_types: List[TrafficType]

    risk_level: Priority

    zone_constraint: ConditionAction

    rationale: str = Field(description="Visual, textual, or baseline rationale for this zone.")

class HighLevelBehaviour(BaseModel):
    reasoning: List[str] = Field(
        min_length=1,
        max_length=5,
        description="Step-by-step reasoning supporting the intent and semantic zones.",
    )

    reflection: MemoryReflection

    route_intent: Action
    chosen_intent: Action

    active_conflict_zones: List[SemanticZone]

    monitor_zones: List[SemanticZone]

    def to_string(self, *, include_reflection: bool = True) -> str:
        """Render to a human-readable string.

        Args:
            include_reflection: When False the REFLECTION block is omitted.
                Pass False when formatting stored memories for VLM prompt
                injection so the model sees only the decision output.
        """
        lines: List[str] = []

        if include_reflection:
            lines.append("REFLECTION:")
            lines.append("- Similarities:")
            for s in self.reflection.retrieved_similarities:
                lines.append(f"\t{s}")

            lines.append("- Differences:")
            for d in self.reflection.critical_differences:
                lines.append(f"\t{d}")

            lines.append("- Reflection Delta:")
            for r_d in self.reflection.extrapolated_risk:
                lines.append(f"\t{r_d}")

            lines.append("")

        lines.append("ROUTE INTENT:")
        lines.append(self.route_intent.value)
        lines.append("")

        lines.append("CHOSEN INTENT:")
        lines.append(self.chosen_intent.value)
        lines.append("")

        lines.append("MONITOR ZONES:")
        if not self.monitor_zones:
            lines.append("- None")
        else:
            for z in self.monitor_zones:
                lines.append(f"- [{z.region.upper()}] risk={z.risk_level}, constraint={z.zone_constraint.value}, entity_types={z.entity_types}, traffic={z.traffic_types}")
                lines.append(f"\t{z.rationale}")

        lines.append("")
        lines.append("ACTIVE CONFLICT ZONES:")
        if not self.active_conflict_zones:
            lines.append("- None")
        else:
            for z in self.active_conflict_zones:
                lines.append(f"- [{z.region.upper()}] risk={z.risk_level}, constraint={z.zone_constraint.value}, entity_types={z.entity_types}, traffic={z.traffic_types}")
                lines.append(f"\t{z.rationale}")

        lines.append("")
        lines.append("REASONING:")
        for r in self.reasoning:
            lines.append(f" - {r}")

        return "\n".join(lines)
