# Model Registry and Instantiator
import base64
import os
import pathlib
from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import Any, Iterable, List, Mapping, Protocol, Sequence

from .message_api import *
from .api_clients import OpenAIClient, OpenRouterClient

class ModelCatalogue:
    _catalogue : Mapping[str, Sequence[str]] = {
        "openai": ["openai/gpt-4.1-2025-04-14"],
        "qwen": ["qwen/qwen2.5-vl-72b-instruct:free"],
        "google": ["google/gemini-2.5-flash"],
        "anthropic": ["anthropic/claude-3-7-sonnet-20250219"],
    }

    @classmethod
    def validate(cls, model_name: str):
        provider = model_name.split("/", 1)[0]
        if provider not in cls._catalogue:
            raise ValueError(
                f"Unknown provider '{provider}'. Valid providers: {list(cls._catalogue)}"
            )
        if model_name not in cls._catalogue[provider]:
            raise ValueError(
                f"Unknown model '{model_name}' for provider '{provider}'. "
                f"Choices: {cls._catalogue[provider]}"
            )

    @classmethod
    def providers(cls) -> Sequence[str]:
        return tuple(cls._catalogue)

    @classmethod
    def models(cls, provider: str) -> Sequence[str]:
        return tuple(cls._catalogue[provider])

class VLMAgent:
    def __init__(self, model_name: str, **kwargs: Any):
        # ModelCatalogue.validate(model_name)

        self.model_name = model_name
        self.client = OpenRouterClient(self.model_name)
        # self.client = OpenAIClient(self.model_name)

        self.temperature = kwargs.get("temperature", 0.0)
        self.max_output_tokens = kwargs.get("max_output_tokens", 500)

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

    def send_message(self, messages: List[Message]) -> str:
        if not messages:
            raise ValueError("Messages cannot be empty.")

        return self.client.send_message(messages, temperature=self.temperature, max_output_tokens=self.max_output_tokens)