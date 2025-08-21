import os

from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain.docstore.document import Document

from .planner_hint_parser import PlanHintParser

class PlannerMemory:
    def __init__(self):
        ws_dir = Path(os.environ['WORK_DIR'])
        self.rag_dir = ws_dir.joinpath('team_code/scene_analyzer/rag_utils/planner_memory')
        db_path = self.rag_dir.joinpath('db/chroma_planner_mem/')

        self.embedding = OpenCLIPEmbeddings(model_name="ViT-B-32", checkpoint="laion2b_s34b_b79k")

        # Always open the store
        self.store = Chroma(
            embedding_function=self.embedding,
            persist_directory=str(db_path)
        )

        # Load seed memories once (when empty)
        count = self.store._collection.count()
        if count == 0:
            self._load_base_memories()

        count = self.store._collection.count()
        print(f"========== Loaded {db_path} Memory. Database has {count} items. ==========")

    def _load_base_memories(self):
        base_hints_path = self.rag_dir.joinpath('planning_hints.txt')
        loader = TextLoader(file_path=str(base_hints_path))
        docs = loader.load()  # typically returns a single Document for the whole file
        if not docs:
            raise FileNotFoundError(f"No content loaded from {base_hints_path}")

        parsed_docs = PlanHintParser.parse_memories_from_text(docs[0].page_content)
        if not parsed_docs:
            raise ValueError(f"Could not parse any memories from {base_hints_path}")

        self.store.add_documents(parsed_docs)
        self.store.persist()

    def retrieve_memories(
        self,
        query : str,
        k : int = 4
    ) -> str:
        few_shot_mems : List[Document] = self.store.similarity_search(query, k=k)
        return self._build_few_shot(few_shot_mems)

    def _build_few_shot(self, retrieved: List[Document]) -> str:
        """
        Convert retrieved docs into a compact few-shot block for your prompt.
        Uses the faithful plan_text from metadata to avoid re-rendering.
        """
        blocks = []
        for d in retrieved:
            meta = d.metadata
            mem_id = meta.get("memory_id", "?")
            plan_text = (meta.get("plan_text") or "").strip()
            reasoning = (meta.get("reasoning") or "").strip()

            block = [
                f"# Memory {mem_id}",
                "Scenario:",
                d.page_content.strip(),
                "Planner Action:",
                plan_text,
            ]
            if reasoning:
                block += ["Reasoning:", reasoning]
            blocks.append("\n".join(block))
        return "\n\n".join(blocks)

    # def retriveMemory(self, driving_scenario: EnvScenario, frame_id: int, top_k: int = 5):
    #     if self.encode_type == 'sce_encode':
    #         pass
    #     elif self.encode_type == 'sce_language':
    #         query_scenario = driving_scenario.describe(frame_id)
    #         similarity_results = self.scenario_memory.similarity_search_with_score(
    #             query_scenario, k=top_k)
    #         fewshot_results = []
    #         for idx in range(0, len(similarity_results)):
    #             # print(f"similarity score: {similarity_results[idx][1]}")
    #             fewshot_results.append(similarity_results[idx][0].metadata)
    #     return fewshot_results

    # def addMemory(self, sce_descrip: str, human_question: str, response: str, action: int, sce: EnvScenario = None, comments: str = ""):
    #     if self.encode_type == 'sce_encode':
    #         pass
    #     elif self.encode_type == 'sce_language':
    #         sce_descrip = sce_descrip.replace("'", '')
    #     # https://docs.trychroma.com/usage-guide#using-where-filters
    #     get_results = self.scenario_memory._collection.get(
    #         where_document={
    #             "$contains": sce_descrip
    #         }
    #     )
    #     # print("get_results: ", get_results)

    #     if len(get_results['ids']) > 0:
    #         # already have one
    #         id = get_results['ids'][0]
    #         self.scenario_memory._collection.update(
    #             ids=id, metadatas={"human_question": human_question,
    #                                'LLM_response': response, 'action': action, 'comments': comments}
    #         )
    #         print("Modify a memory item. Now the database has ", len(
    #             self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
    #     else:
    #         doc = Document(
    #             page_content=sce_descrip,
    #             metadata={"human_question": human_question,
    #                       'LLM_response': response, 'action': action, 'comments': comments}
    #         )
    #         id = self.scenario_memory.add_documents([doc])
    #         print("Add a memory item. Now the database has ", len(
    #             self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")

    # def deleteMemory(self, ids):
    #     self.scenario_memory._collection.delete(ids=ids)
    #     print("Delete", len(ids), "memory items. Now the database has ", len(
    #         self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")

    # def combineMemory(self, other_memory):
    #     other_documents = other_memory.scenario_memory._collection.get(
    #         include=['documents', 'metadatas', 'embeddings'])
    #     current_documents = self.scenario_memory._collection.get(
    #         include=['documents', 'metadatas', 'embeddings'])
    #     for i in range(0, len(other_documents['embeddings'])):
    #         if other_documents['embeddings'][i] in current_documents['embeddings']:
    #             print("Already have one memory item, skip.")
    #         else:
    #             self.scenario_memory._collection.add(
    #                 embeddings=other_documents['embeddings'][i],
    #                 metadatas=other_documents['metadatas'][i],
    #                 documents=other_documents['documents'][i],
    #                 ids=other_documents['ids'][i]
    #             )
    #     print("Merge complete. Now the database has ", len(
    #         self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
