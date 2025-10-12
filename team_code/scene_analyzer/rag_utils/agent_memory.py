import os
import chromadb
import numpy as np

from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from PIL import Image

class AgentMemory:
    def __init__(self, db_path=None) -> None:
        # Initialize the ChromaDB client
        db_path = os.path.join(
            './agent_memory_db', 'chroma_5_shot_100_mem/') if db_path is None else db_path
        self.chroma_client = chromadb.PersistentClient(
            path=db_path
        )

        # Set embedding function for image and text
        self.embedding_func = OpenCLIPEmbeddingFunction(
            model_name='ViT-g-14',
            checkpoint="laion2b_s34b_b88k",
            device='cuda'
        )

        # Create image collection
        self.agent_memory_img = self.chroma_client.get_or_create_collection(
            name="agent_memory_img",
            embedding_function=self.embedding_func,
            metadata={"hnsw:space": "cosine"}
        )

        # Create text collection
        self.agent_memory_text = self.chroma_client.get_or_create_collection(
            name="agent_memory_text",
            embedding_function=self.embedding_func,
            metadata={"hnsw:space": "cosine"}
        )
        
        # print("==========Loaded ",db_path," Memory, Now the database has ", len(
        #     self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.==========")

    def retrieve_memory(
            self, 
            img_bev: Image.Image, 
            scene_description: str, 
            top_k: int = 10,
            alpha: float = 0.5,
            beta: float = 0.5,
            top_n: int = 5):
        """
        Retrieve relevant memories based on image and text inputs, combining results 
        from both modalities using weighted scoring.
        Args:
            img_bev (Image.Image): The bird's-eye view image to be converted into an embedding 
                for similarity search.
            scene_description (str): A textual description of the scene to be used for 
                similarity search.
            top_k (int, optional): The number of top results to retrieve from each modality. 
                Defaults to 10.
            alpha (float, optional): Weighting factor for the image-based similarity score 
                when combining results. Defaults to 0.5.
            beta (float, optional): Weighting factor for the text-based similarity score 
                when combining results. Defaults to 0.5.
            top_n (int, optional): The number of top combined results to return. Defaults to 5.
        Returns:
            List[str]: A list of memory IDs corresponding to the top combined results.
            np.ndarray: An array of combined similarity scores for the top results.
        """
        # Encode image and text inputs
        img_query_embedding = self.embedding_func(img_bev)
        text_query_embedding = self.embedding_func(scene_description)

        # Query similar images from the image collection
        img_results = self.agent_memory_img.query(
            query_embeddings=[img_query_embedding],
            n_results=top_k,
            include=['embeddings', 'metadatas', 'distances']
        )
        img_mem_ids   = [metadatas["mem_id"] for metadatas in img_results["metadatas"][0]]
        img_dists  = np.array(img_results["distances"][0], dtype=np.float32)

        # Query similar texts from the text collection
        text_results = self.agent_memory_text.get(
            where={"mem_id": {"$in": img_mem_ids}},
            include=['embeddings', 'metadatas']
        )
        text_embeddings = np.array(text_results['embeddings'][0], dtype=np.float32)

        # Text cosine similarity
        text_similarity = text_embeddings @ text_query_embedding.T
        # text_similarity = text_similarity / (np.linalg.norm(text_embeddings, axis=1) * np.linalg.norm(text_query_embedding))
        text_dists = (1 - text_similarity)

        # Combine image and text results using weighted scoring
        combined_scores = alpha * img_dists + beta * text_dists
        ordered_indices = combined_scores.argsort()[:top_n]

        return [img_mem_ids[idx] for idx in ordered_indices], combined_scores[ordered_indices]
        
    def retriveMemory(self, driving_scenario: EnvScenario, frame_id: int, top_k: int = 5):
        if self.encode_type == 'sce_encode':
            pass
        elif self.encode_type == 'sce_language':
            query_scenario = driving_scenario.describe(frame_id)
            similarity_results = self.scenario_memory.similarity_search_with_score(
                query_scenario, k=top_k)
            fewshot_results = []
            for idx in range(0, len(similarity_results)):
                # print(f"similarity score: {similarity_results[idx][1]}")
                fewshot_results.append(similarity_results[idx][0].metadata)
        return fewshot_results

    def addMemory(self, sce_descrip: str, human_question: str, response: str, action: int, sce: EnvScenario = None, comments: str = ""):
        if self.encode_type == 'sce_encode':
            pass
        elif self.encode_type == 'sce_language':
            sce_descrip = sce_descrip.replace("'", '')
        # https://docs.trychroma.com/usage-guide#using-where-filters
        get_results = self.scenario_memory._collection.get(
            where_document={
                "$contains": sce_descrip
            }
        )
        # print("get_results: ", get_results)

        if len(get_results['ids']) > 0:
            # already have one
            id = get_results['ids'][0]
            self.scenario_memory._collection.update(
                ids=id, metadatas={"human_question": human_question,
                                   'LLM_response': response, 'action': action, 'comments': comments}
            )
            print("Modify a memory item. Now the database has ", len(
                self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
        else:
            doc = Document(
                page_content=sce_descrip,
                metadata={"human_question": human_question,
                          'LLM_response': response, 'action': action, 'comments': comments}
            )
            id = self.scenario_memory.add_documents([doc])
            print("Add a memory item. Now the database has ", len(
                self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")

    def deleteMemory(self, ids):
        self.scenario_memory._collection.delete(ids=ids)
        print("Delete", len(ids), "memory items. Now the database has ", len(
            self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")

    def combineMemory(self, other_memory):
        other_documents = other_memory.scenario_memory._collection.get(
            include=['documents', 'metadatas', 'embeddings'])
        current_documents = self.scenario_memory._collection.get(
            include=['documents', 'metadatas', 'embeddings'])
        for i in range(0, len(other_documents['embeddings'])):
            if other_documents['embeddings'][i] in current_documents['embeddings']:
                print("Already have one memory item, skip.")
            else:
                self.scenario_memory._collection.add(
                    embeddings=other_documents['embeddings'][i],
                    metadatas=other_documents['metadatas'][i],
                    documents=other_documents['documents'][i],
                    ids=other_documents['ids'][i]
                )
        print("Merge complete. Now the database has ", len(
            self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")


if __name__ == "__main__":
    pass