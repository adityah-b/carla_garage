import os
import cv2
import uuid
import json
import time
import tempfile

import numpy as np

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_core.documents import Document

from typing import Dict, Any, Optional, Tuple

from team_code.scene_analyzer.rag_utils.planner_memory.dual_stage.planner_hint_parser import PlanHintParser
from team_code.scene_analyzer.parsers.ego_plan_pydantic_models import EgoPlan

class PlannerMemory:
    def __init__(self):
        ws_dir = Path(os.environ['WORK_DIR'])
        self.rag_dir = ws_dir.joinpath('team_code/scene_analyzer/rag_utils/planner_memory')
        db_path = self.rag_dir.joinpath('db/chroma_planner_mem/')

        self.embedding = OpenCLIPEmbeddings(model_name="ViT-B-32", checkpoint="laion2b_s34b_b79k")

        self.images_dir = self.rag_dir.joinpath('db/images')
        self.images_dir.mkdir(parents=True, exist_ok=True)

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
        parsed = PlanHintParser.parse_memories()
        if not parsed:
            raise ValueError("No seed memories returned from PlanHintParser")

        docs: List[Document] = []
        for memory_id, hl_beh, ego_plan in parsed:
            driving_skill = hl_beh.primary_intent.value
            added_at_iso = datetime.now(timezone.utc).isoformat()
            added_at_ts = int(time.time() * 1000)

            payload = self._build_payload(
                episode_id=memory_id,
                driving_skill=driving_skill,
                added_at_iso=added_at_iso,
                added_at_ts=added_at_ts,
                hl_behaviour=hl_beh,
                ego_plan=ego_plan,
            )

            # Embedding text: ONLY the HighLevelBehaviour semantic string.
            # The EgoPlan is payload only — it must not pollute the vector space.
            embedding_text = hl_beh.to_string()

            # Full memory text for RAG display (includes both hl_beh and ego_plan)
            display_text = self.format_memory_text(payload)

            metadata: Dict[str, Any] = {
                "episode_id": memory_id,
                "driving_skill": driving_skill,
                "added_at_iso": added_at_iso,
                "added_at_ts": added_at_ts,
                "hl_primary_intent": driving_skill,
                "ego_action": getattr(
                    getattr(ego_plan, "action", None),
                    "value",
                    str(getattr(ego_plan, "action", "")),
                ),
                "payload_json": json.dumps(payload, ensure_ascii=False),
                "display_text": display_text,
                "is_seed": True,
            }

            # page_content is what gets embedded — use the hl_beh semantic string
            docs.append(Document(page_content=embedding_text, metadata=metadata))

        self.store.add_documents(docs, ids=[memory_id for memory_id, _, _ in parsed])
        self.store.persist()

    def num_items(self) -> int:
        return int(self.store._collection.count())

    def _save_image(self, episode_id: str, image: np.ndarray) -> str:
        path = str(self.images_dir / f"{episode_id}.png")
        cv2.imwrite(path, image)
        return path

    def _compute_query_embedding(
        self,
        text: str,
        image: Optional[np.ndarray] = None,
        text_weight: float = 0.5,
    ) -> List[float]:
        """Compute embedding from text and optional image via weighted average in CLIP space."""
        text_emb = np.array(self.embedding.embed_query(text))

        if image is None:
            return text_emb.tolist()

        # OpenCLIPEmbeddings.embed_image expects file paths
        fd, temp_path = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        try:
            cv2.imwrite(temp_path, image)
            image_emb = np.array(self.embedding.embed_image([temp_path])[0])
        finally:
            os.unlink(temp_path)

        fused = text_weight * text_emb + (1.0 - text_weight) * image_emb
        norm = np.linalg.norm(fused)
        if norm > 1e-12:
            fused = fused / norm
        return fused.tolist()

    def _doc_to_payload(self, doc: Document) -> Optional[Dict[str, Any]]:
        """
        Prefer metadata['payload_json'] so we can always reconstruct even if page_content changes.
        Returns None for older/seed docs that don't store payload_json.
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
        *,
        query_embedding: Optional[List[float]] = None,
    ) -> Tuple[Optional[Document], Optional[float]]:
        """
        Returns (best_doc, best_similarity) in the given skill bucket.

        If query_embedding is provided, searches by vector; otherwise by text.
        Similarity is in [0,1] when available; otherwise a pseudo-relevance computed from distance.
        """
        if query_embedding is not None:
            # Vector-based search with pre-computed (fused) embedding
            if hasattr(self.store, "similarity_search_by_vector_with_relevance_scores"):
                try:
                    pairs = self.store.similarity_search_by_vector_with_relevance_scores(
                        embedding=query_embedding,
                        k=k,
                        filter={"driving_skill": driving_skill},
                    )
                    if not pairs:
                        return None, None
                    best_doc, best_rel = max(pairs, key=lambda p: p[1])
                    return best_doc, float(best_rel)
                except Exception:
                    pass

        # Preferred: text-based relevance scores (0..1 higher is more similar)
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
        hl_behaviour : Any,
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
          - caps long lists to avoid bloating RAG
        """
        episode_id = payload.get("episode_id", "unknown")
        driving_skill = payload.get("driving_skill", "unknown")

        hl = payload.get("high_level_behaviour", {})
        ep = payload.get("ego_plan", {})

        hl_intent = hl.get("primary_intent", "")
        # Support both pipeline schemas:
        # single_stage has target_speed_limit; dual_stage has drivable_space_status
        if "target_speed_limit" in hl:
            hl_space = f"speed_limit={hl['target_speed_limit']}"
        else:
            hl_space = hl.get("drivable_space_status", "")
        hl_reasoning = hl.get("reasoning", [])
        conflict_zones = hl.get("conflict_zones", [])

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

        def format_conflict_zones(zones: List[Any]) -> str:
            if not zones:
                return "- (none)"
            lines: List[str] = []
            for z in cap(zones):
                if isinstance(z, dict):
                    region = z.get("region", "")
                    actors = z.get("entity_types", [])
                    traffic = z.get("traffic_types", [])
                    risk = z.get("risk_level", "")
                    action = z.get("suggested_action", "")
                    desc = z.get("description", "")
                    action_str = f" action={action}" if action else ""
                    lines.append(f"- [{region}] actors={actors} traffic={traffic} risk={risk}{action_str}: {desc}")
                else:
                    lines.append(f"- {z}")
            return "\n".join(lines)

        def format_conditions(conds: List[Any]) -> str:
            if not conds:
                return "- (none)"
            lines: List[str] = []
            for c in cap(conds):
                if isinstance(c, dict):
                    ca = c.get("condition_action", "")
                    target = c.get("target", {})
                    region = target.get("region", "")
                    actor = target.get("actor_type", "")
                    traffic = target.get("traffic_type", "")
                    priority = c.get("priority", "medium")
                    lines.append(f"- {ca} [{region}] actor={actor} traffic={traffic} priority={priority}")
                else:
                    lines.append(f"- {c}")
            return "\n".join(lines)

        text = (
            f"MEMORY_ENTRY\n"
            f"Episode: {episode_id}\n"
            f"DrivingSkill: {driving_skill}\n\n"
            f"HighLevelBehaviour\n"
            f"- primary_intent: {hl_intent}\n"
            f"- drivable_space: {hl_space}\n"
            f"- conflict_zones:\n{format_conflict_zones(conflict_zones)}\n"
            f"- reasoning:\n{bullets(hl_reasoning)}\n\n"
            f"EgoPlan\n"
            f"- action: {plan_action}\n"
            f"- conditions:\n{format_conditions(conditions)}\n"
            f"- reasoning:\n{bullets(plan_reasoning)}\n"
        )

        return text

    def add_memory(
        self,
        hl_behaviour : Any,
        ego_plan : EgoPlan,
        *,
        image: Optional[np.ndarray] = None,
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
        episode_id = str(uuid.uuid4())
        added_at_iso = datetime.now(timezone.utc).isoformat()
        added_at_ts = int(time.time() * 1000)  # epoch ms

        driving_skill = hl_behaviour.primary_intent.value

        payload = self._build_payload(
            episode_id=episode_id,
            driving_skill=driving_skill,
            added_at_iso=added_at_iso,
            added_at_ts=added_at_ts,
            hl_behaviour=hl_behaviour,
            ego_plan=ego_plan,
        )

        # Embedding text: ONLY the HighLevelBehaviour semantic string.
        # EgoPlan is NOT embedded — it is payload only.
        embedding_text = hl_behaviour.to_string()

        # Compute embedding (fused text+image when image is available)
        query_embedding = self._compute_query_embedding(embedding_text, image)

        # Full display text for RAG
        display_text = self.format_memory_text(
            payload,
            max_list_items=max_list_items_for_storage,
        )

        # ---- Novelty gate: compare against existing episodes in the same skill ----
        best_doc, best_sim = self._best_similarity_within_skill(
            query_text=embedding_text,
            driving_skill=driving_skill,
            k=ncheck,
            query_embedding=query_embedding,
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

        # Save image to disk if provided
        image_path = ""
        if image is not None:
            image_path = self._save_image(episode_id, image)

        metadata: Dict[str, Any] = {
            "episode_id": episode_id,
            "driving_skill": driving_skill,
            "added_at_iso": added_at_iso,
            "added_at_ts": added_at_ts,
            "hl_primary_intent": driving_skill,
            "ego_action": getattr(getattr(ego_plan, "action", None), "value", str(getattr(ego_plan, "action", ""))),
            "payload_json": json.dumps(payload, ensure_ascii=False),
            "display_text": display_text,
            "image_path": image_path,
            "is_seed": False,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        # Insert with pre-computed embedding via low-level API
        def _upsert():
            self.store._collection.add(
                ids=[episode_id],
                embeddings=[query_embedding],
                metadatas=[metadata],
                documents=[embedding_text],
            )

        try:
            _upsert()
        except Exception:
            try:
                self._delete_ids([episode_id])
            except Exception:
                pass
            _upsert()

        return episode_id

    def retrieve_memories(
        self,
        hl_behaviour : Any,
        num_entries : int = 3,
        *,
        image: Optional[np.ndarray] = None,
        max_list_items_for_rag: Optional[int] = 10,
        backfill_global: bool = True,
    ) -> List[Document]:
        """
        Retrieve relevant memories for RAG.

        Two-Level Filtering:
          Level 1 (Metadata): Filter by driving_skill == primary_intent.value
          Level 2 (Semantic):  Embed hl_behaviour.to_string() (+ optional image) and vector-search

        The EgoPlan is NOT part of the embedding — it is retrieved as payload only.
        """
        if num_entries <= 0:
            return []

        driving_skill = hl_behaviour.primary_intent.value

        # Query text = the semantic scene context (conflict zones + intent)
        query_text = hl_behaviour.to_string()

        # 1) Level 1 + Level 2: within skill bucket, semantic search
        if image is not None:
            query_embedding = self._compute_query_embedding(query_text, image)
            docs: List[Document] = self.store.similarity_search_by_vector(
                embedding=query_embedding,
                k=num_entries,
                filter={"driving_skill": driving_skill},
            )
        else:
            docs: List[Document] = self.store.similarity_search(
                query=query_text,
                k=num_entries,
                filter={"driving_skill": driving_skill},
            )

        # 2) Re-render each doc via display_text or format_memory_text for consistent RAG
        rendered: List[Document] = []
        for d in docs:
            # Prefer stored display_text, fall back to re-rendering from payload
            md = d.metadata or {}
            display = md.get("display_text")
            if display:
                rendered.append(Document(page_content=display, metadata=deepcopy(md)))
                continue

            payload = self._doc_to_payload(d)
            if payload is None:
                rendered.append(d)
                continue

            text = self.format_memory_text(
                payload,
                max_list_items=max_list_items_for_rag,
            )
            rendered.append(Document(page_content=text, metadata=deepcopy(md)))

        return rendered

    def retrieve_memories_by_text(
        self,
        query_text: str,
        num_entries: int = 3,
        *,
        image: Optional[np.ndarray] = None,
    ) -> List[Document]:
        """
        Retrieve memories by text/image similarity without a driving-skill filter.

        Used in the NS pipeline's Stage 1, where the intent is not yet known.
        """
        if num_entries <= 0:
            return []

        if image is not None:
            query_embedding = self._compute_query_embedding(query_text, image)
            docs: List[Document] = self.store.similarity_search_by_vector(
                embedding=query_embedding,
                k=num_entries,
            )
        else:
            docs: List[Document] = self.store.similarity_search(
                query=query_text,
                k=num_entries,
            )

        rendered: List[Document] = []
        for d in docs:
            md = d.metadata or {}
            display = md.get("display_text")
            if display:
                rendered.append(Document(page_content=display, metadata=deepcopy(md)))
                continue
            payload = self._doc_to_payload(d)
            if payload is None:
                rendered.append(d)
                continue
            text = self.format_memory_text(payload)
            rendered.append(Document(page_content=text, metadata=deepcopy(md)))

        return rendered
