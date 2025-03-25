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
        self.model_name = "gemini-1.5-flash"
        self.config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
            response_schema=HighLevelCommand
        )

        # Initialize the LLM with the system prompt
        self._initialize_llm()

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