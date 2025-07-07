import numpy as np
import io

from google import genai
from google.genai import types
from PIL import Image
from pydantic import TypeAdapter

from llm_output_schema import *

class SceneInterpreter:
    def __init__(self, system_prompt: str, temperature=0.4, max_output_tokens=500, api_key = None):
        """
        Initialize the SceneInterpreter with the LLM endpoint, API key, and system prompt.
        :param llm_endpoint: URL of the LLM API endpoint.
        :param api_key: API key for authentication.
        :param system_prompt: System prompt to initialize the LLM.
        """
        # self.api_key = api_key
        self.api_key = "AIzaSyBkaG1tw1zURzhNjT5rvdy-2yZhpPQ4V1Y"
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        # self.model_name = "gemini-1.5-flash"
        # self.model_name = "gemini-2.5-flash-preview-04-17"
        self.model_name = "gemini-2.0-flash"
        self.config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
            response_schema=HighLevelCommand
        )

        # Initialize the LLM with the system prompt
        self._initialize_llm()

    def summarize_scene(self, textual_description: str, image: np.ndarray) -> SceneDescription:
        """
        Summarizes the scene description and image into a SceneDescription object.
        :param scene_description: Textual description of the scene.
        :param image: Image of the scene as a numpy array.
        :return: SceneDescription object containing key actors and their types.
        """
        try:
            # Convert the numpy array to a PIL image
            pil_image = Image.fromarray(np.uint8(image), mode='RGB')
            contents = [pil_image, textual_description]

            prompt = f"""
You are an expert autonomous driving assistant. You will be given a RGB images of the Bird's Eye View (BEV) and front-view with respect to the ego vehicle. Alongside this information, you will be provided with a textual representation of the scene metadata. Given this information, provide a concise, natural language summary of the current driving context. Your response will be a SceneDescription object that includes the following fields:

road_description: Description of the road network and its conditions.
traffic_description: Description of the traffic conditions in the scene.
static_objects_and_obstacles_description: Description of static objects and obstacles in the scene.
ego_vehicle_description: Description of the ego vehicle and its state.
key_actors: List of key actors in the scene as a KeyActor object.
reasoning: Reasoning for the chosen key actors.
"""
            config = types.GenerateContentConfig(
                system_instruction=prompt,
                temperature=0.0,
                max_output_tokens=self.max_output_tokens,
                response_mime_type="application/json",
                response_schema=SceneDescription
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
            print(f"Scene Summary: {response.text}")
            scene_description_adapter = TypeAdapter(SceneDescription)
            scene_desc = scene_description_adapter.validate_python(response.parsed)
            return scene_desc

        except Exception as e:
            return {"error": str(e)}

#     def predict_intention(self, scene_description: SceneDescription, image: np.ndarray) -> NPCIntention:
#         """
#         Predicts the intention of the non-player characters (NPCs) in the scene.
#         :param scene_description: SceneDescription object containing key actors and their types.
#         :param image: Image of the scene as a numpy array.
#         :return: NPCIntention object containing predicted intentions of NPCs.
#         """
#         try:
#             # Convert the numpy array to a PIL image
#             pil_image = Image.fromarray(np.uint8(image), mode='RGB')
#             key_actors =
# #             scene_prompt = f"""
# # Scene Description:
# #     Road Description: {scene_description.road_description}
# #     Traffic Description: {scene_description.traffic_description}
# #     Static Objects and Obstacles: {scene_description.static_objects_and_obstacles_description}
# #     Ego Vehicle State: {scene_description.ego_vehicle_description}

#             contents = [pil_image, scene_description]


    def _initialize_llm(self):
        """
        Sends the system prompt to the LLM to initialize it.
        """
        try:
            self.client = genai.Client(api_key=self.api_key)
            ret = self.client.models.update(
                model=self.model_name,
                config=self.config,
            )
            print(f'Response: {ret}')
            print("LLM initialized successfully with the system prompt.")
        except Exception as e:
            print(f"Error initializing LLM: {e}")

    def _validate_output(self, output):
        pass

    def run_step(self, scene_description, image) -> HighLevelCommand:
        try:
            # Convert the numpy array to a PIL image
            pil_image = Image.fromarray(np.uint8(image), mode='RGB')
            contents = [pil_image, scene_description]

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=self.config
            )
            print(f"Response: {response.text}")
            high_level_cmds_adapter = TypeAdapter(HighLevelCommand)
            high_level_cmds = high_level_cmds_adapter.validate_python(response.parsed)
            return high_level_cmds

        except Exception as e:
            return {"error": str(e)}