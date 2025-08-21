import re

from typing import Any, Dict, List, Optional
from langchain.docstore.document import Document

class PlanHintParser:
    @staticmethod
    def _try_parse_value(raw: str) -> Any:
        """Parse numeric when possible; otherwise return trimmed string (macros preserved)."""
        s = raw.strip()
        # allow speed_limit-2.0 etc. → keep as string macro
        try:
            return float(s)
        except ValueError:
            return s

    @staticmethod
    def parse_memories_from_text(text: str) -> List[Document]:
        docs: List[Document] = []

        # Split the text by memory headers first
        memory_blocks = re.split(r'(?=^# Memory \d+:)', text, flags=re.MULTILINE)

        # Remove empty blocks
        memory_blocks = [block.strip() for block in memory_blocks if block.strip()]

        for block in memory_blocks:
            # Skip if this doesn't look like a memory block
            if not block.startswith('# Memory'):
                continue

            # Extract memory ID
            id_match = re.match(r'# Memory (\d+):', block)
            if not id_match:
                continue
            mem_id = int(id_match.group(1))

            # Find Scenario section
            scenario_match = re.search(r'Scenario:\s*\n(.*?)(?=\nPlanner Action:)', block, re.DOTALL)
            if not scenario_match:
                print(f"Warning: Could not find scenario in memory {mem_id}")
                continue
            scenario = scenario_match.group(1).strip()

            # Find Planner Action section
            plan_match = re.search(r'Planner Action:\s*\n(.*?)(?=\nReasoning:|\n# Memory|\Z)', block, re.DOTALL)
            if not plan_match:
                print(f"Warning: Could not find planner action in memory {mem_id}")
                continue
            plan = plan_match.group(1).strip()

            # Find optional Reasoning section
            reasoning_match = re.search(r'Reasoning:\s*\n(.*?)(?=\n# Memory|\Z)', block, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

            metadata = {
                "memory_id": mem_id,
                "plan_text": plan,
                "reasoning": reasoning,
            }

            # IMPORTANT: only scenario goes into page_content (for retrieval)
            docs.append(Document(page_content=scenario, metadata=metadata))
            print(f"Parsed Memory {mem_id}: '{scenario[:50]}...'")

        return docs