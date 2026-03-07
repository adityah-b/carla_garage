import os
import uuid
import json
import time

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_core.documents import Document

from typing import Dict, Any

from team_code.scene_analyzer.rag_utils.planner_memory.planner_hint_parser import PlanHintParser
from team_code.scene_analyzer.parsers.hl_beh_pydantic_models import *
from team_code.scene_analyzer.parsers.ego_plan_pydantic_models import *

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

    def num_items(self) -> int:
        return int(self.store._collection.count())

    def _doc_to_payload(self, doc: Document) -> Optional[Dict[str, Any]]:
        """
        Prefer metadata['payload_json'] so we can always reconstruct even if page_content changes.
        Returns None for older/seed docs that don’t store payload_json.
        """
        md = doc.metadata or {}
        pj = md.get("payload_json")
        if isinstance(pj, str) and pj.strip():
            try:
                return json.loads(pj)
            except Exception:
                return None
        return None

    def _collection_get(self, *, where: Optional[Dict[str, Any]] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Returns (ids, metadatas) with best-effort compatibility across Chroma versions.
        """
        try:
            if where is None:
                res = self.store._collection.get(include=["metadatas"])
            else:
                res = self.store._collection.get(where=where, include=["metadatas"])
            ids = list(res.get("ids", []))
            metas = list(res.get("metadatas", []))
            return ids, metas
        except Exception:
            # fallback: fetch all and filter manually
            try:
                res = self.store._collection.get(include=["metadatas"])
                all_ids = list(res.get("ids", []))
                all_metas = list(res.get("metadatas", []))
                if where is None:
                    return all_ids, all_metas
                ids, metas = [], []
                for _id, m in zip(all_ids, all_metas):
                    m = m or {}
                    ok = True
                    for k, v in where.items():
                        if m.get(k) != v:
                            ok = False
                            break
                    if ok:
                        ids.append(_id)
                        metas.append(m)
                return ids, metas
            except Exception:
                return [], []

    def _delete_ids(self, ids: List[str]) -> None:
        if not ids:
            return
        # langchain vectorstore API
        if hasattr(self.store, "delete"):
            try:
                self.store.delete(ids=ids)
                return
            except Exception:
                pass
        # chroma internal
        try:
            self.store._collection.delete(ids=ids)
        except Exception:
            pass

    def _best_similarity_within_skill(
        self,
        query_text: str,
        driving_skill: str,
        k: int,
    ) -> Tuple[Optional[Document], Optional[float]]:
        """
        Returns (best_doc, best_similarity) in the given skill bucket.

        Similarity is in [0,1] when available; otherwise a pseudo-relevance computed from distance.
        """
        # Preferred: relevance scores (0..1 higher is more similar)
        if hasattr(self.store, "similarity_search_with_relevance_scores"):
            try:
                pairs = self.store.similarity_search_with_relevance_scores(
                    query=query_text,
                    k=k,
                    filter={"driving_skill": driving_skill},
                )
                if not pairs:
                    return None, None
                best_doc, best_rel = max(pairs, key=lambda p: p[1])
                return best_doc, float(best_rel)
            except Exception:
                pass

        # Fallback: score often is a distance (lower is more similar)
        if hasattr(self.store, "similarity_search_with_score"):
            try:
                pairs = self.store.similarity_search_with_score(
                    query=query_text,
                    k=k,
                    filter={"driving_skill": driving_skill},
                )
                if not pairs:
                    return None, None
                best_doc, best_dist = min(pairs, key=lambda p: p[1])
                try:
                    dist = float(best_dist)
                    # pseudo relevance in (0,1], smaller dist => higher relevance
                    rel = 1.0 / (1.0 + max(dist, 0.0))
                    return best_doc, rel
                except Exception:
                    return best_doc, None
            except Exception:
                pass

        return None, None

    def _prune_to_caps(
        self,
        *,
        max_total_entries: Optional[int],
        max_entries_per_skill: Optional[int],
        driving_skill_for_skill_cap: Optional[str],
        protect_seed: bool,
        reserve_slots: int,
    ) -> None:
        """
        Prunes oldest entries to satisfy caps.
        Uses metadata['added_at_ts'] for ordering. No timestamp parsing required.
        """
        # Total cap
        if max_total_entries is not None:
            ids, metas = self._collection_get(where=None)
            limit = max_total_entries - reserve_slots
            if ids and len(ids) > limit:
                candidates: List[Tuple[str, Dict[str, Any]]] = []
                for _id, m in zip(ids, metas):
                    m = m or {}
                    if protect_seed and m.get("is_seed", False):
                        continue
                    candidates.append((_id, m))

                # oldest first
                candidates.sort(key=lambda x: int((x[1] or {}).get("added_at_ts", -1)))
                overflow = len(ids) - limit
                to_delete = [candidates[i][0] for i in range(min(overflow, len(candidates)))]
                self._delete_ids(to_delete)

        # Per-skill cap
        if max_entries_per_skill is not None and driving_skill_for_skill_cap is not None:
            ids, metas = self._collection_get(where={"driving_skill": driving_skill_for_skill_cap})
            limit = max_entries_per_skill - reserve_slots
            if ids and len(ids) > limit:
                candidates: List[Tuple[str, Dict[str, Any]]] = []
                for _id, m in zip(ids, metas):
                    m = m or {}
                    if protect_seed and m.get("is_seed", False):
                        continue
                    candidates.append((_id, m))

                candidates.sort(key=lambda x: int((x[1] or {}).get("added_at_ts", -1)))
                overflow = len(ids) - limit
                to_delete = [candidates[i][0] for i in range(min(overflow, len(candidates)))]
                self._delete_ids(to_delete)

    def _build_payload(
        self,
        episode_id : int,
        driving_skill : str,
        added_at_iso : str,
        added_at_ts : int,
        hl_behaviour : HighLevelBehaviour,
        ego_plan : EgoPlan,
        *,
        extra : Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        hl = hl_behaviour.model_dump()
        ep = ego_plan.model_dump()

        payload = {
            "episode_id": episode_id,
            "driving_skill": driving_skill,
            "added_at_iso": added_at_iso,
            "added_at_ts": added_at_ts,
            "high_level_behaviour": hl,
            "ego_plan": ep,
        }

        if extra:
            payload.update(extra)

        return payload

    def format_memory_text(
        self,
        payload: Dict[str, Any],
        *,
        max_list_items: Optional[int] = None,
    ) -> str:
        """
        Canonical renderer for:
          - RAG input to the LLM
          - memory inspection/debugging

        max_list_items:
          - caps long lists (actors/objects/obstacles) to avoid bloating RAG
        """
        episode_id = payload.get("episode_id", "unknown")
        driving_skill = payload.get("driving_skill", "unknown")

        hl = payload.get("high_level_behaviour", {})
        ep = payload.get("ego_plan", {})

        hl_next = hl.get("next_action", "")
        hl_reasoning = hl.get("reasoning", [])

        plan_action = ep.get("action", "")
        plan_reasoning = ep.get("reasoning", [])
        conditions = ep.get("conditions", [])

        def cap(xs: List[Any]) -> List[Any]:
            if max_list_items is None:
                return xs
            return xs[: max_list_items]

        def bullets(items: List[Any], prefix: str = "\t- ") -> str:
            if not items:
                return f"{prefix}(none)"
            out = []
            for it in items:
                if isinstance(it, dict):
                    out.append(prefix + json.dumps(it, ensure_ascii=False))
                else:
                    out.append(prefix + str(it))
            return "\n".join(out)

        def format_conditions(conds: List[Any]) -> str:
            if not conds:
                return "- (none)"
            lines: List[str] = []
            for c in conds:
                if isinstance(c, dict):
                    ca = c.get("condition_action", "")
                    obj_type = c.get("obj_type", "")
                    tid = c.get("id", "")
                    ttype = c.get("traffic_type", "")
                    imp = c.get("importance", 1.0)
                    lines.append(f"- {ca} {obj_type} id={tid} traffic_type={ttype} importance={imp}")
                else:
                    lines.append(f"- {c}")
            return "\n".join(lines)

        text = (
            f"MEMORY_ENTRY\n"
            f"Episode: {episode_id}\n"
            f"DrivingSkill: {driving_skill}\n\n"
            f"HighLevelBehaviour\n"
            f"- next_action: {hl_next}\n"
            f"- reasoning:\n{bullets(hl_reasoning)}\n\n"
            f"EgoPlan\n"
            f"- action: {plan_action}\n"
            f"- conditions:\n{format_conditions(conditions)}\n"
            f"- reasoning:\n{bullets(plan_reasoning)}\n"
        )

        return text

    def add_memory(
        self,
        hl_behaviour : HighLevelBehaviour,
        ego_plan : EgoPlan,
        *,
        extra_metadata: Optional[Dict[str, Any]] = None,
        # override defaults (optional)
        max_similarity_to_existing : float = 0.8,
        num_neighbors_check : int = 5,
        max_total_entries : int = 50,
        max_entries_per_skill : int = 5,
        protect_seed : bool = True,
        # formatting knobs
        max_list_items_for_storage: Optional[int] = None,
    ) -> str:
        """
        Adds a memory only if sufficiently novel within the same driving_skill bucket.

        If the new memory is too similar to an existing one in that skill, it is NOT inserted,
        and the existing episode_id is returned when available.

        Caps are enforced before insertion (pruning oldest to make room).
        """
        # defaults
        max_similarity = max_similarity_to_existing
        ncheck = num_neighbors_check
        total_cap = max_total_entries
        skill_cap = max_entries_per_skill
        protect = protect_seed

        # ids + timestamps (no parsing required later)
        episode_id = int(uuid.uuid4())
        added_at_iso = datetime.now(timezone.utc).isoformat()
        added_at_ts = int(time.time() * 1000)  # epoch ms

        driving_skill = hl_behaviour.next_action

        payload = self._build_payload(
            episode_id=episode_id,
            driving_skill=driving_skill,
            added_at_iso=added_at_iso,
            added_at_ts=added_at_ts,
            hl_behaviour=hl_behaviour,
            ego_plan=ego_plan,
        )

        # Organized doc content used for BOTH embedding + readability
        page_content = self.format_memory_text(
            payload,
            max_list_items=max_list_items_for_storage,
        )

        # ---- Novelty gate: compare against existing episodes in the same skill ----
        best_doc, best_sim = self._best_similarity_within_skill(
            query_text=page_content,
            driving_skill=driving_skill,
            k=ncheck,
        )

        if best_sim is not None and best_sim >= max_similarity:
            existing_id = (best_doc.metadata or {}).get("episode_id") if best_doc else None
            # If we can identify the existing id, return it and skip insert.
            if existing_id:
                return existing_id
            # If we can't identify it (e.g., seed/older entries), fall through and insert.

        # ---- Enforce caps (make room for this insertion) ----
        self._prune_to_caps(
            max_total_entries=total_cap,
            max_entries_per_skill=skill_cap,
            driving_skill_for_skill_cap=driving_skill if skill_cap is not None else None,
            protect_seed=protect,
            reserve_slots=1,
        )

        metadata: Dict[str, Any] = {
            "episode_id": episode_id,
            "driving_skill": driving_skill,
            "added_at_iso": added_at_iso,
            "added_at_ts": added_at_ts,
            "hl_next_action": str(hl_behaviour.next_action),
            "ego_action": getattr(getattr(ego_plan, "action", None), "value", str(getattr(ego_plan, "action", ""))),
            "payload_json": json.dumps(payload, ensure_ascii=False),
            "is_seed": False,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        doc = Document(page_content=page_content, metadata=metadata)

        # Upsert-ish behavior: replace if episode_id already exists
        try:
            self.store.add_documents([doc], ids=[episode_id])
        except Exception:
            try:
                self._delete_ids([episode_id])
            except Exception:
                pass
            self.store.add_documents([doc], ids=[episode_id])

        return episode_id

    def retrieve_memories(
        self,
        hl_behaviour : HighLevelBehaviour,
        num_entries : int = 3,
        *,
        max_list_items_for_rag: Optional[int] = 10,
        backfill_global: bool = True,
    ) -> List[Document]:
        """
        Retrieve relevant memories for RAG.

        Steps:
          1) driving_skill bucket = normalize(hl_behaviour.next_action)
          2) similarity query = "\\n".join(hl_behaviour.reasoning) (fallback to driving_skill if empty)
          3) similarity_search within the same skill bucket
          4) optionally backfill globally if bucket is sparse
          5) re-render each doc via format_memory_text(payload) when payload_json exists
        """
        if num_entries <= 0:
            return []

        driving_skill = hl_behaviour.next_action
        reasoning_lines = list(hl_behaviour.reasoning or [])
        query_text = "\n".join(reasoning_lines).strip() or driving_skill

        # 1) within skill bucket
        docs: List[Document] = self.store.similarity_search(
            query=query_text,
            k=num_entries,
            filter={"driving_skill": driving_skill},
        )

        # 3) canonical re-render (consistent RAG formatting)
        rendered: List[Document] = []
        for d in docs:
            payload = self._doc_to_payload(d)
            if payload is None:
                # seed/older docs might not have payload_json
                rendered.append(d)
                continue

            text = self.format_memory_text(
                payload,
                max_list_items=max_list_items_for_rag,
            )
            rendered.append(Document(page_content=text, metadata=deepcopy(d.metadata or {})))

        return rendered