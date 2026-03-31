from typing import List, Tuple

from team_code.scene_analyzer.rag_utils.planner_memory.single_stage.planning_hints import SEED_MEMORIES


class PlanHintParser:
    @staticmethod
    def parse_memories() -> List[Tuple[str, str, str]]:
        """Return a list of (memory_id, scene_text, behavior_string) triples."""
        results: List[Tuple[str, str, str]] = []

        for key, entry in SEED_MEMORIES.items():
            scene_text: str = entry["scene_text"]
            behavior_string: str = entry["high_level_behaviour"]
            results.append((key, scene_text, behavior_string))
            print(f"Parsed seed memory '{key}'")

        return results
