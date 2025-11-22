# Model Registry and Instantiator
import base64
import os
import pathlib
from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import Any, Iterable, List, Dict
from pydantic import BaseModel

from .message_api import *
from .api_clients import *

class ModelCatalogue:
    _clients : Dict[str, APIClient] = {
        "openai" : OpenAIClient,
        "openrouter" : OpenRouterClient,
        "vllm" : VLLMClient
    }

    _catalogue : Dict[str, List[str]] = {
        "openai" : ["gpt-4.1-mini-2025-04-14"],
        "openrouter" : ["openai/gpt-4.1-2025-04-14", "qwen/qwen2.5-vl-72b-instruct:free", "google/gemini-2.5-flash", "anthropic/claude-3-7-sonnet-20250219"],
        "vllm" : ["Qwen/Qwen2.5-VL-72B-Instruct-AWQ", "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"],
    }

    @staticmethod
    def validate(provider : str, model_name: str):
        if provider not in ModelCatalogue._catalogue:
            raise ValueError(
                f"Unknown provider '{provider}'. Valid providers: {list(ModelCatalogue._catalogue)}"
            )
        if model_name not in ModelCatalogue._catalogue[provider]:
            raise ValueError(
                f"Unknown model '{model_name}' for provider '{provider}'. "
                f"Choices: {ModelCatalogue._catalogue[provider]}"
            )

    @staticmethod
    def providers() -> List[str]:
        return tuple(ModelCatalogue._catalogue)

    @staticmethod
    def models(provider: str) -> List[str]:
        return tuple(ModelCatalogue._catalogue[provider])

    @staticmethod
    def client(provider : str) -> APIClient:
        return ModelCatalogue._clients[provider]

class VLMAgent:
    def __init__(self, provider : str, model_name: str, **kwargs: Any):
        ModelCatalogue.validate(provider, model_name)

        self.model_name = model_name
        self.provider = provider

        self.client = ModelCatalogue.client(provider)(self.model_name)

        self.temperature = kwargs.get("temperature", 0.0)
        self.max_output_tokens = kwargs.get("max_output_tokens", 2048)
        print(f'max_output_tokens: {self.max_output_tokens}')

    def create_user_message(self, text : str = None, image : Union[Path, np.ndarray] = None) -> Message:
        builder = MessageBuilder(MessageRole.USER)

        if text is not None:
            builder.add_text_content(text)

        if image is not None:
            builder.add_image_content(image)

        return builder.build_message()

    def create_system_message(self, text: str) -> Message:
        builder = MessageBuilder(MessageRole.DEVELOPER)
        # builder = MessageBuilder(MessageRole.SYSTEM)
        builder.add_text_content(text)
        return builder.build_message()

    def send_message(self, messages: List[Message], text_format : BaseModel = None) -> str:
        if not messages:
            raise ValueError("Messages cannot be empty.")

        return self.client.send_message(messages, text_format=text_format, temperature=self.temperature, max_output_tokens=self.max_output_tokens)