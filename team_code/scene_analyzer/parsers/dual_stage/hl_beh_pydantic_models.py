from pydantic import BaseModel, Field
from typing import List, Literal

from team_code.scene_analyzer.parsers.base_pydantic_models import Action, MemoryReflection


class ConflictZone(BaseModel):
    region: Literal["ego", "left", "right", "oncoming", "crossing", "any"]
    entity_types: List[
        Literal[
            "vehicle", "pedestrian", "cyclist", "obstacle",
            "emergency_vehicle", "traffic_light", "stop_sign", "any",
        ]
    ]
    traffic_types: List[
        Literal["oncoming", "crossing", "leading", "trailing", "traffic_object"]
    ]
    risk_level: Literal["low", "medium", "high"]
    description: str = Field(description="Brief explanation of the conflict physics.")


class HighLevelBehaviour(BaseModel):
    reflection: MemoryReflection
    primary_intent: Action
    conflict_zones: List[ConflictZone]
    drivable_space_status: Literal["open", "blocked"]
    reasoning: List[str] = Field(
        min_length=1,
        max_length=5,
        description="Step-by-step reasoning supporting the intent and conflict zones.",
    )

    def to_string(self) -> str:
        """This string is exclusively what gets embedded for the vector search."""
        lines = [
            f"INTENT: {self.primary_intent.value}",
            f"SPACE: {self.drivable_space_status}\n",
            "CONFLICT ZONES:",
        ]
        if not self.conflict_zones:
            lines.append("- None")
        else:
            for zone in self.conflict_zones:
                lines.append(
                    f"- {zone.region.upper()}: Risk is {zone.risk_level}. {zone.description}"
                )
        return "\n".join(lines)
