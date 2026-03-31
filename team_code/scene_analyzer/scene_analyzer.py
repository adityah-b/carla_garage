import importlib

import numpy as np

from typing import List, Union, Optional, Any
from pathlib import Path

from langchain_core.documents import Document

from .llm_agents import VLMAgent
from .sys_prompts import SysPrompts
from .rag_utils.planner_memory.planner_memory import PlannerMemory
from .rag_utils.planner_memory.single_stage.planner_memory import SingleStagePlannerMemory

from .parsers.ego_plan_pydantic_models import EgoPlan, PlanState

_PIPELINE_MODULES = {
    "dual_stage":   "team_code.scene_analyzer.parsers.dual_stage.hl_beh_pydantic_models",
    "single_stage": "team_code.scene_analyzer.parsers.single_stage.hl_beh_pydantic_models",
}

_SINGLE_STAGE_TRANSLATOR_MODULE = "team_code.scene_analyzer.parsers.single_stage.deterministic_translator"


class SceneAnalyzer(VLMAgent):
    def __init__(
        self,
        provider: str,
        model_name: str,
        pipeline_mode: str = "single_stage",
        **kwargs,
    ):
        super().__init__(provider, model_name, **kwargs)

        if pipeline_mode not in _PIPELINE_MODULES:
            raise ValueError(
                f"Unknown pipeline_mode '{pipeline_mode}'. "
                f"Valid options: {list(_PIPELINE_MODULES)}"
            )

        self.pipeline_mode = pipeline_mode
        self.sys_prompts = SysPrompts()
        if pipeline_mode == "single_stage":
            self.planning_memory = SingleStagePlannerMemory()
        else:
            self.planning_memory = PlannerMemory()

        # Dynamically import the active pipeline's HighLevelBehaviour class
        mod = importlib.import_module(_PIPELINE_MODULES[pipeline_mode])
        self._HighLevelBehaviour = mod.HighLevelBehaviour

        # single_stage pipeline uses the deterministic translator instead of a second VLM call
        if pipeline_mode == "single_stage":
            translator_mod = importlib.import_module(_SINGLE_STAGE_TRANSLATOR_MODULE)
            self._translate = translator_mod.translate
        else:
            self._translate = None

    # ── Stage 1 ───────────────────────────────────────────────────────────────

    def get_high_level_behaviour(
        self,
        scene_summary: str,
        scene_image: Optional[np.ndarray] = None,
        prev_plan_state: Optional[PlanState] = None,
    ) -> Any:
        """
        Run Stage 1: produce a HighLevelBehaviour for the active pipeline.

        prev_plan_state: when provided, must be a FAILED plan — its action,
        conditions, and failure reason are injected into the prompt so the VLM
        knows what was tried and why it did not work.

        dual_stage:   plain scene text → VLM → HighLevelBehaviour
        single_stage: scene text + retrieved memories → VLM → HighLevelBehaviour
                      (memories are injected so MemoryReflection can be filled)
        """
        if self.pipeline_mode == "single_stage":
            system_instruction = self.sys_prompts.get_single_stage_hl_beh_prompt()
            user_text = self._build_single_stage_prompt(
                text=scene_summary,
                image=scene_image,
                prev_plan_state=prev_plan_state,
            )
        else:
            system_instruction = self.sys_prompts.get_high_level_behaviour_prompt()
            prev_plan_text = ""
            if prev_plan_state is not None:
                prev_plan_text = (
                    f"## Previous Failed Plan\n"
                    f"{prev_plan_state.to_string()}\n\n"
                )
            user_text = f"{prev_plan_text}## Current Scenario\n{scene_summary}\n"

        system_message = self.create_system_message(text=system_instruction)

        print(f'\n\nHigh Level Planning Prompt\n\n')
        print(user_text)

        user_message = self.create_user_message(text=user_text, image=scene_image)
        messages = [system_message, user_message]

        response = self.send_message(messages, text_format=self._HighLevelBehaviour)
        return response.parsed

    def _build_single_stage_prompt(
        self,
        text: str,
        image: Optional[np.ndarray] = None,
        *,
        dense_text: Optional[str] = None,
        prev_plan_state: Optional[PlanState] = None,
    ) -> str:
        """Retrieve memories via two-stage retrieval and prepend to user message.

        Args:
            text:       Scene summary used as the Stage-1 CLIP query text.
            image:      Optional ego image fused into the CLIP query.
            dense_text: Optional full-detail scene text for Stage-2 dense
                        reranking.  Defaults to `text` when not provided.
        """
        memory_docs: List[Document] = self.planning_memory.retrieve_memories_by_text(
            query_text=text,
            num_entries=3,
            image=image,
            dense_text=dense_text,
        )

        memory_text = ""
        if memory_docs:
            entries = "\n\n".join(
                f"--- MEMORY {i + 1} ---\n{doc.page_content}"
                for i, doc in enumerate(memory_docs)
            )
            memory_text = (
                f"## Planning Memories\n"
                f"**Not all memories may be relevant, use only those that apply.**\n"
                f"{entries}\n\n"
            )

        prev_plan_text = ""
        if prev_plan_state is not None:
            prev_plan_text = (
                f"## Previous Failed Plan\n"
                f"{prev_plan_state.to_string()}\n\n"
            )

        return f"{memory_text}{prev_plan_text}## Current Scenario\n{text}\n"

    # ── Stage 2 ───────────────────────────────────────────────────────────────

    def get_ego_plan(
        self,
        text: str,
        hl_beh: Any,
        image: Union[Path, np.ndarray] = None,
        prev_plan: Optional[PlanState] = None,
    ) -> EgoPlan:
        """
        Run Stage 2: produce an EgoPlan from the HighLevelBehaviour.

        dual_stage:   memory-augmented prompt → VLM → EgoPlan
        single_stage: deterministic rule-based translation (no VLM call)
        """
        if self.pipeline_mode == "single_stage":
            return self._translate(hl_beh)

        system_instruction = self.sys_prompts.get_plan_gen_prompt()
        system_message = self.create_system_message(text=system_instruction)

        memory_image = image if isinstance(image, np.ndarray) else None
        plan_prompt = self._build_planning_prompt(text, hl_beh, prev_plan, image=memory_image)

        print(f'\n\nPlanning Prompt\n\n')
        print(plan_prompt)

        user_message = self.create_user_message(text=plan_prompt, image=image)
        messages = [system_message, user_message]

        response = self.send_message(messages, text_format=EgoPlan)
        return response.parsed

    # ── Memory ────────────────────────────────────────────────────────────────

    def log_episode(self, prev_plan_state: PlanState) -> None:
        """Log a cleanly-completed plan execution to memory.

        Scene text and image are read from the plan state itself — they were
        captured at the moment the plan was created, so the memory embedding
        is keyed on the scene that drove the planning decision.
        """
        if self.pipeline_mode == "single_stage":
            episode_id = self.planning_memory.add_memory(
                prev_plan_state.hl_beh,
                scene_text=prev_plan_state.scene_text,
                image=prev_plan_state.scene_image,
            )
        else:
            episode_id = self.planning_memory.add_memory(
                hl_behaviour=prev_plan_state.hl_beh,
                ego_plan=prev_plan_state.plan,
                image=prev_plan_state.scene_image,
            )
        print(f'\n\nLogged episode to memory: {episode_id}\n\n')

    # ── VLM pipeline helpers ──────────────────────────────────────────────────

    def _build_planning_prompt(
        self,
        text: str,
        hl_beh: Any,
        prev_plan: Optional[PlanState] = None,
        *,
        image: Optional[np.ndarray] = None,
    ) -> str:
        memory_docs: List[Document] = self.planning_memory.retrieve_memories(
            hl_behaviour=hl_beh,
            num_entries=3,
            image=image,
        )
        memory_text = ""
        if memory_docs:
            entries = "\n\n".join(doc.page_content for doc in memory_docs)
            memory_text = (
                f"## Planning Memories\n"
                f"- Not all memories may be relevant — use only what applies.\n"
                f"{entries}\n\n"
            )

        prev_plan_text = ""
        if prev_plan:
            prev_plan_text = f"## Previous Ego Plan\n{prev_plan.to_string()}\n\n"

        hl_beh_text = f"## High-Level Behaviour\n{hl_beh.to_string()}\n\n"

        return f"{memory_text}{prev_plan_text}{hl_beh_text}## Current Scenario\n{text}\n"
