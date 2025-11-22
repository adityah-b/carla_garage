from abc import ABC, abstractmethod
from openai import OpenAI
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel

from .message_api import *
from .error_exceptions import APIClientError

class APIClient(ABC):
    @abstractmethod
    def send_message(self, messages: List[Message], **kwargs) -> str:
        """Send a message to the API and return the response."""
        pass

    @abstractmethod
    def _format_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Format messages for the API request."""
        pass

    @abstractmethod
    def _format_content(self, content: MessageContent) -> Dict[str, str]:
        """Format content for the API request."""
        pass

class OpenAIClient(APIClient):
    def __init__(self, model_name: str):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise APIClientError("OPENAI_API_KEY environment variable is not set.")

        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        formatted_messages = []
        for message in messages:
            formatted_message = {
                "role": message.role.value,
                "content": message.content if isinstance(message.content, str) else [self._format_content(content) for content in message.content]
            }
            formatted_messages.append(formatted_message)
        return formatted_messages

    def _format_content(self, content: MessageContent) -> Dict[str, str]:
        if content.type == ContentType.TEXT:
            return {"type": "input_text", "text": content.data}
        elif content.type == ContentType.IMAGE:
            return {"type": "input_image", "image_url": content.data}
        else:
            raise APIClientError(f"Unsupported content type: {content.type}")

    def send_message(self, messages: List[Message], **kwargs):
        if not messages:
            raise APIClientError("Messages cannot be empty.")

        formatted_messages = self._format_messages(messages)
        text_format = kwargs.get("text_format", None)
        if text_format:
            response = self.client.responses.parse(
                model="gpt-4.1-mini-2025-04-14",
                input=formatted_messages,
                temperature=0.0,
                max_output_tokens=1024,
                text_format=text_format
            )
        else:
            response = self.client.responses.create(
                model="gpt-4.1-mini-2025-04-14",
                input=formatted_messages,
                temperature=0.0,
                max_output_tokens=512,
            )

        print(f'\n\nUSAGE\n\n')
        print(f'{response.usage}')

        return response

class OpenRouterClient(APIClient):
    def __init__(self, model_name: str):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"

        if not self.api_key:
            raise APIClientError("OPENROUTER_API_KEY environment variable is not set.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model_name = model_name

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        formatted_messages = []
        for message in messages:
            formatted_message = {
                "role": message.role.value,
                "content": message.content if isinstance(message.content, str) else [self._format_content(content) for content in message.content]
            }
            formatted_messages.append(formatted_message)
        return formatted_messages


    def _format_content(self, content: MessageContent) -> Dict[str, str]:
        if content.type == ContentType.TEXT:
            return {"type": "text", "text": content.data}
        elif content.type == ContentType.IMAGE:
            return {
                "type": "image_url",
                "image_url": {
                    "url" : content.data
                }
            }
        else:
            raise APIClientError(f"Unsupported content type: {content.type}")

    def send_message(self, messages: List[Message], **kwargs) -> str:
        if not messages:
            raise APIClientError("Messages cannot be empty.")

        formatted_messages = self._format_messages(messages)

        response = self.client.chat.completions.create(
            model=self.model_name,
            extra_body={},
            messages=formatted_messages,
            temperature=0.2,
            max_completion_tokens=1024,
        )

        # print(f'\n\nRAW OUTPUT\n\n')
        # print(f'{repr(response.choices[0].message.content)}')
        # print(f'\n\nUSAGE\n\n')
        # print(f'{response.usage}')
        # print(f'\n\nSTOP REASON\n\n')
        # print(f'{response.choices[0].finish_reason}')
        # print(f'\n\nJSON DUMP\n\n')
        # print(f'{response.model_dump_json(indent=2)}')
        return response.choices[0].message.content

class VLLMClient(APIClient):
    def __init__(self, model_name : str):
        self.api_key = os.getenv("VLLM_API_KEY", "EMPTY")
        # self.base_url = "http://0.0.0.0:8000/v1"
        # self.base_url = "http://127.0.0.1:8000/v1"
        self.base_url = "http://192.168.42.200:8000/v1"

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model_name = model_name

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        formatted_messages = []
        for message in messages:
            formatted_message = {
                "role": message.role.value,
                "content": message.content
                if isinstance(message.content, str)
                else [self._format_content(content) for content in message.content],
            }
            formatted_messages.append(formatted_message)
        return formatted_messages

    def _format_content(self, content: MessageContent) -> Dict[str, str]:
        if content.type == ContentType.TEXT:
            return {
                "type": "text",
                "text": content.data
            }
        elif content.type == ContentType.IMAGE:
            return {
                "type": "image_url",
                "image_url": {
                    "url": content.data
                }
            }
        else:
            raise APIClientError(f"Unsupported content type: {content.type}")

    def send_message(self, messages: List[Message], **kwargs) -> str:
        if not messages:
            raise APIClientError("Messages cannot be empty.")

        formatted_messages = self._format_messages(messages)

        text_format = kwargs.get("text_format", None)
        if text_format:
            response = self.client.chat.completions.parse(
                model=self.model_name,
                messages=formatted_messages,
                temperature=kwargs.get("temperature", 0.0),
                max_completion_tokens=kwargs.get("max_output_tokens", 4096),
                response_format=text_format
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                temperature=kwargs.get("temperature", 0.0),
                max_completion_tokens=kwargs.get("max_output_tokens", 4096),
            )

        print(f'\n\nUSAGE\n\n')
        print(f'{response.usage}')

        return response.choices[0].message
