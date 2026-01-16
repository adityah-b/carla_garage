import numpy as np

from typing import Union, Optional, Dict
from pathlib import Path

from .llm_agents import VLMAgent
from .sys_prompts import SysPrompts
from .rag_utils.planner_memory.planner_memory import PlannerMemory

from .parsers.hl_beh_pydantic_models import HighLevelBehaviour
from .parsers.ego_plan_pydantic_models import EgoPlan, PlanExecution

class SceneAnalyzer(VLMAgent):
    def __init__(
        self,
        provider : str,
        model_name: str,
        **kwargs
    ):
        super().__init__(provider, model_name, **kwargs)

        self.sys_prompts = SysPrompts()
        self.planning_memory = PlannerMemory()

    def get_high_level_behaviour(
        self,
        text : str,
        image: Union[Path, np.ndarray] = None,
    ) -> HighLevelBehaviour:
        system_instruction = self.sys_prompts.get_high_level_behaviour_prompt()
        system_message = self.create_system_message(text=system_instruction)

        # TODO: Incorporate scenario memory
        print(f'\n\nHigh Level Planning Prompt\n\n')
        print(f'{text}')
        user_message = self.create_user_message(text=text, image=image)
        messages = [system_message, user_message]

        response = self.send_message(messages, text_format=HighLevelBehaviour)
        # print(f'\n\nHIGH LEVEL BEHAVIOUR RAW RESPONSE\n\n')
        # print(f'{response.output_text}')
        hl_beh : HighLevelBehaviour = response.parsed

        return hl_beh

    def get_ego_plan(
        self,
        text: str,
        image: Union[Path, np.ndarray] = None,
        prev_plan : Optional[PlanExecution] = None,
    ) -> EgoPlan:
        system_instruction = self.sys_prompts.get_plan_gen_prompt()

        # print(f'\n\nSystem Planning Prompt\n\n')
        # print(f'{system_instruction}')

        system_message = self.create_system_message(text=system_instruction)
        plan_prompt = self._build_planning_prompt(text, prev_plan)

        print(f'\n\nPlanning Prompt\n\n')
        print(f'{plan_prompt}')
        user_message = self.create_user_message(text=plan_prompt, image=image)

        messages = [system_message, user_message]

        response = self.send_message(messages, text_format=EgoPlan)
        # print(f'\n\nEGO PLAN RAW RESPONSE\n\n')
        # print(f'{response}')

        ego_plan : EgoPlan = response.parsed

        return ego_plan

    def _build_planning_prompt(
        self,
        text : str,
        prev_plan: Optional[PlanExecution] = None,
    ) -> str:
        few_shot_results = self.planning_memory.retrieve_memories(text, k=2)
#         plan_prompt = f"""
# ## Planning Memories
# - !!! Not all memories are relevant to the task. Select the most relevant ones. !!!
# - **Memory Entries**:
# {few_shot_results}

# ## Current Scenario
# {text}
# """
        prev_plan_text = ""
        if prev_plan:
            prev_plan_text = f"""## Previous Ego Plan\n{prev_plan.to_string()}\n\n"""

        plan_prompt = f"""{prev_plan_text}## Current Scenario
{text}
"""
        return plan_prompt
