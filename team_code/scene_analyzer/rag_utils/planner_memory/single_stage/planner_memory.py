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
from typing import List, Dict, Any, Optional, Tuple

from langchain_community.vectorstores import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from team_code.scene_analyzer.rag_utils.planner_memory.single_stage.planner_hint_parser import PlanHintParser


class SingleStagePlannerMemory:
    """
    Episodic memory for the single-stage planning pipeline.

    Architecture — Retrieve-and-Rerank:
      Stage 1  ChromaDB + OpenCLIP  Fused (scene_text + ego image) → top K candidates.
      Stage 2  SentenceTransformer  Dense cosine similarity reranking → top N results.

    Each stored memory contains:
      embedding      Fused CLIP vector  (scene_text + optional image)
      document       scene_text         (the formatted scene summary, stored for debugging)
      behavior_string hl_beh.to_string(include_reflection=False) injected into VLM prompts
      dense_vector   Pre-computed SBERT embedding of scene_text (fast Stage-2 reranking)

    Memories are keyed only by ID — no skill-based bucketing.
    """

    def __init__(self):
        ws_dir = Path(os.environ['WORK_DIR'])
        self.rag_dir = ws_dir.joinpath(
            'team_code/scene_analyzer/rag_utils/planner_memory/single_stage'
        )
        db_path = self.rag_dir.joinpath('db/chroma_planner_mem/')

        self.clip_embedding = OpenCLIPEmbeddings(
            model_name="ViT-B-32", checkpoint="laion2b_s34b_b79k"
        )
        self.dense_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.images_dir = self.rag_dir.joinpath('db/images')
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.store = Chroma(
            embedding_function=self.clip_embedding,
            persist_directory=str(db_path),
        )

        if self.store._collection.count() == 0:
            self._load_base_memories()

        count = self.store._collection.count()
        print(
            f"========== [SingleStage] Loaded memory at {db_path}. "
            f"Database has {count} items. =========="
        )

    # ── Seed loading ──────────────────────────────────────────────────────────

    def _load_base_memories(self) -> None:
        parsed = PlanHintParser.parse_memories()
        if not parsed:
            raise ValueError("No seed memories returned from single-stage PlanHintParser")

        docs: List[Document] = []
        ids: List[str] = []

        for memory_id, scene_text, behavior_string in parsed:
            added_at_iso = datetime.now(timezone.utc).isoformat()
            added_at_ts = int(time.time() * 1000)

            dense_vec = self.dense_model.encode(scene_text).tolist()

            metadata: Dict[str, Any] = {
                "episode_id": memory_id,
                "added_at_iso": added_at_iso,
                "added_at_ts": added_at_ts,
                "behavior_string": behavior_string,
                "dense_vector": json.dumps(dense_vec),
                "is_seed": True,
            }

            # page_content = scene_text (what gets embedded and shown in debug)
            docs.append(Document(page_content=scene_text, metadata=metadata))
            ids.append(memory_id)

        # Use low-level API so we can supply pre-computed embeddings
        clip_embs = [self._compute_clip_embedding(d.page_content) for d in docs]
        self.store._collection.add(
            ids=ids,
            embeddings=clip_embs,
            metadatas=[d.metadata for d in docs],
            documents=[d.page_content for d in docs],
        )
        self.store.persist()

    # ── Embedding helpers ─────────────────────────────────────────────────────

    def _compute_clip_embedding(
        self,
        text: str,
        image: Optional[np.ndarray] = None,
        text_weight: float = 0.5,
    ) -> List[float]:
        """Fused CLIP embedding: weighted average of text and optional image vectors."""
        text_emb = np.array(self.clip_embedding.embed_query(text))

        if image is None:
            return text_emb.tolist()

        fd, tmp = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        try:
            cv2.imwrite(tmp, image)
            img_emb = np.array(self.clip_embedding.embed_image([tmp])[0])
        finally:
            os.unlink(tmp)

        fused = text_weight * text_emb + (1.0 - text_weight) * img_emb
        norm = np.linalg.norm(fused)
        if norm > 1e-12:
            fused = fused / norm
        return fused.tolist()

    # ── Stage 2: dense reranker ───────────────────────────────────────────────

    def _rerank_by_dense(
        self,
        docs: List[Document],
        dense_query: str,
        top_n: int,
    ) -> List[Document]:
        """
        Re-score `docs` by cosine similarity between a live SBERT embedding of
        `dense_query` and the pre-computed dense_vector stored in each document's
        metadata, then return the top_n highest-scoring documents.
        """
        if not docs:
            return []

        v_query = self.dense_model.encode(dense_query)
        q_norm = float(np.linalg.norm(v_query))

        scored: List[Tuple[float, Document]] = []
        for doc in docs:
            dv_json = (doc.metadata or {}).get("dense_vector")
            if dv_json is None:
                # No pre-computed vector — assign zero score (do not drop)
                scored.append((0.0, doc))
                continue

            v_mem = np.array(json.loads(dv_json))
            m_norm = float(np.linalg.norm(v_mem))
            if q_norm < 1e-12 or m_norm < 1e-12:
                scored.append((0.0, doc))
            else:
                score = float(np.dot(v_query, v_mem) / (q_norm * m_norm))
                doc.metadata["rerank_score"] = score
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_n]]

    # ── Persistence helpers ───────────────────────────────────────────────────

    def _collection_get(
        self, *, where: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        try:
            res = (
                self.store._collection.get(include=["metadatas"])
                if where is None
                else self.store._collection.get(where=where, include=["metadatas"])
            )
            return list(res.get("ids", [])), list(res.get("metadatas", []))
        except Exception:
            try:
                res = self.store._collection.get(include=["metadatas"])
                all_ids = list(res.get("ids", []))
                all_metas = list(res.get("metadatas", []))
                if where is None:
                    return all_ids, all_metas
                ids, metas = [], []
                for _id, m in zip(all_ids, all_metas):
                    m = m or {}
                    if all(m.get(k) == v for k, v in where.items()):
                        ids.append(_id)
                        metas.append(m)
                return ids, metas
            except Exception:
                return [], []

    def _delete_ids(self, ids: List[str]) -> None:
        if not ids:
            return
        if hasattr(self.store, "delete"):
            try:
                self.store.delete(ids=ids)
                return
            except Exception:
                pass
        try:
            self.store._collection.delete(ids=ids)
        except Exception:
            pass

    def _prune_to_total_cap(
        self,
        max_total: int,
        reserve_slots: int = 1,
        protect_seed: bool = True,
    ) -> None:
        ids, metas = self._collection_get()
        limit = max_total - reserve_slots
        if not ids or len(ids) <= limit:
            return
        candidates = [
            (_id, m or {})
            for _id, m in zip(ids, metas)
            if not (protect_seed and (m or {}).get("is_seed", False))
        ]
        candidates.sort(key=lambda x: int(x[1].get("added_at_ts", -1)))
        overflow = len(ids) - limit
        self._delete_ids([candidates[i][0] for i in range(min(overflow, len(candidates)))])

    # ── Public API ────────────────────────────────────────────────────────────

    def num_items(self) -> int:
        return int(self.store._collection.count())

    def add_memory(
        self,
        hl_behaviour: Any,
        *,
        scene_text: Optional[str] = None,
        image: Optional[np.ndarray] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        max_similarity_to_existing: float = 0.8,
        num_neighbors_check: int = 5,
        max_total_entries: int = 50,
        protect_seed: bool = True,
    ) -> str:
        """
        Embed and store a new episodic memory.

        Args:
            hl_behaviour:  Single-stage HighLevelBehaviour produced by the VLM.
            scene_text:    The formatted scene summary used as the Stage 1 CLIP
                           embedding input.  Falls back to hl_behaviour.to_string()
                           when not provided.
            image:         Optional ego camera frame fused into the CLIP embedding.
            max_similarity_to_existing:
                           Skip insertion when the best existing neighbour has
                           cosine similarity >= this threshold.

        Returns:
            episode_id of the inserted (or duplicate-detected) memory.
        """
        episode_id = str(uuid.uuid4())
        added_at_iso = datetime.now(timezone.utc).isoformat()
        added_at_ts = int(time.time() * 1000)

        # Behavior string: the VLM decision without reflection
        behavior_string = hl_behaviour.to_string(include_reflection=False)

        # Embedding text: use the scene summary when available
        embedding_text = scene_text if scene_text else hl_behaviour.to_string(include_reflection=False)

        clip_emb = self._compute_clip_embedding(embedding_text, image)
        dense_vec = self.dense_model.encode(embedding_text).tolist()

        # ── Novelty gate ──────────────────────────────────────────────────────
        try:
            pairs = self.store.similarity_search_by_vector_with_relevance_scores(
                embedding=clip_emb,
                k=num_neighbors_check,
            )
            if pairs:
                best_doc, best_sim = max(pairs, key=lambda p: p[1])
                if best_sim >= max_similarity_to_existing:
                    existing_id = (best_doc.metadata or {}).get("episode_id")
                    if existing_id:
                        return existing_id
        except Exception:
            pass

        # ── Enforce total cap ─────────────────────────────────────────────────
        self._prune_to_total_cap(
            max_total=max_total_entries,
            reserve_slots=1,
            protect_seed=protect_seed,
        )

        # ── Save image ────────────────────────────────────────────────────────
        image_path = ""
        if image is not None:
            image_path = str(self.images_dir / f"{episode_id}.png")
            cv2.imwrite(image_path, image)

        metadata: Dict[str, Any] = {
            "episode_id": episode_id,
            "added_at_iso": added_at_iso,
            "added_at_ts": added_at_ts,
            "behavior_string": behavior_string,
            "dense_vector": json.dumps(dense_vec),
            "image_path": image_path,
            "is_seed": False,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        def _upsert() -> None:
            self.store._collection.add(
                ids=[episode_id],
                embeddings=[clip_emb],
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

    def retrieve_memories_by_text(
        self,
        query_text: str,
        num_entries: int = 3,
        *,
        image: Optional[np.ndarray] = None,
        dense_text: Optional[str] = None,
        top_k: int = 20,
    ) -> List[Document]:
        """
        Two-stage retrieval:
          Stage 1  CLIP multimodal query (query_text + optional image) → top_k docs.
          Stage 2  Dense reranker on dense_text (defaults to query_text) → num_entries docs.

        Returns LangChain Documents whose page_content is the behavior_string to
        be injected into the VLM prompt.

        Args:
            query_text:  Simplified scene summary for the CLIP embedding query.
            num_entries: Number of memories to return after reranking.
            image:       Optional ego image fused into the CLIP query.
            dense_text:  Full-detail scene text for Stage-2 reranking.
                         Defaults to query_text when not provided.
            top_k:       Candidate pool size retrieved in Stage 1.
        """
        if num_entries <= 0:
            return []

        top_k = max(top_k, num_entries)
        clip_emb = self._compute_clip_embedding(query_text, image)

        stage1_docs: List[Document] = self.store.similarity_search_by_vector(
            embedding=clip_emb,
            k=top_k,
        )

        rerank_query = dense_text if dense_text else query_text
        if len(stage1_docs) > num_entries:
            final_docs = self._rerank_by_dense(stage1_docs, rerank_query, num_entries)
        else:
            final_docs = stage1_docs[:num_entries]

        return self._render_docs(final_docs)

    def _render_docs(self, docs: List[Document]) -> List[Document]:
        """Format each document as a scene-text → decision pair for prompt injection."""
        rendered = []
        for d in docs:
            md = deepcopy(d.metadata or {})
            behavior = md.get("behavior_string")
            scene = d.page_content  # always the original scene summary
            if behavior:
                content = f"### Input Scene\n{scene}\n\n### Planning Decision\n{behavior}"
            else:
                content = scene
            rendered.append(Document(page_content=content, metadata=md))
        return rendered
