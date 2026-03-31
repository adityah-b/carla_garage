from typing import List, Tuple

from team_code.scene_analyzer.rag_utils.planner_memory.dual_stage.planning_hints import SEED_MEMORIES
from team_code.scene_analyzer.parsers.dual_stage.hl_beh_pydantic_models import HighLevelBehaviour
from team_code.scene_analyzer.parsers.ego_plan_pydantic_models import EgoPlan


class PlanHintParser:
    @staticmethod
    def parse_memories() -> List[Tuple[str, HighLevelBehaviour, EgoPlan]]:
        results: List[Tuple[str, HighLevelBehaviour, EgoPlan]] = []

        for key, entry in SEED_MEMORIES.items():
            hl_beh = HighLevelBehaviour.model_validate(entry["high_level_behaviour"])
            ego_plan = EgoPlan.model_validate(entry["ego_plan"])
            results.append((key, hl_beh, ego_plan))
            print(f"Parsed seed memory '{key}' (skill='{hl_beh.primary_intent.value}')")

        return results
