import numpy as np
import cv2
import os
import base64
import json

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
from enum import Enum, auto
from pathlib import Path

from .error_exceptions import ImageEncoderError, MessageBuilderError

class MessageRole(Enum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"

class ContentType(Enum):
    TEXT = auto()
    IMAGE = auto()

@dataclass(frozen=True, slots=True)
class MessageContent:
    type: ContentType
    data: str

@dataclass(frozen=True, slots=True)
class Message:
    role : MessageRole
    content : Union[str, List[MessageContent]]

class ImageEncoder:
    @staticmethod
    def encode_image(image: Union[Path, np.ndarray]) -> str:
        def resize_image(img: np.ndarray, scale: float = 0.3) -> np.ndarray:
            height, width = img.shape[:2]
            new_size = (int(width * scale), int(height * scale))
            return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

        if isinstance(image, Path):
            try:
                if not image.exists():
                    raise ImageEncoderError(f"Image file {image} does not exist.")

                if image.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    raise ImageEncoderError(f"Unsupported image format: {image.suffix}. Supported formats are .jpg, .jpeg, and .png.")

                img = cv2.imread(str(image))
                if img is None:
                    raise ImageEncoderError(f"Failed to read image file {image} using OpenCV.")

                resized_img = resize_image(img)

                _, buffer = cv2.imencode(image.suffix.lower(), resized_img)
                b64_image = base64.b64encode(buffer).decode('utf-8')
                mime_type = 'jpeg' if image.suffix.lower() in ['.jpg', '.jpeg'] else 'png'
                b64_image_str = f"data:image/{mime_type};base64,{b64_image}"

                # with open(image, "rb") as img_file:
                #     b64_image = base64.b64encode(img_file.read()).decode('utf-8')
                #     b64_image_str = f"data:image/{image.suffix[1:]};base64,{b64_image}"

            except IOError as e:
                raise ImageEncoderError(f"Error reading image file {image}: {e}")

        elif isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] not in [3, 4]:
                raise ImageEncoderError("Invalid image array shape. Expected a 3D array with 3 (RGB) or 4 (RGBA) channels.")

            resized_img = resize_image(image)

            _, buffer = cv2.imencode('.png', resized_img)
            b64_image = base64.b64encode(buffer).decode('utf-8')
            b64_image_str = f"data:image/png;base64,{b64_image}"

        else:
            raise TypeError("Image must be a numpy array or a Path object pointing to an image file.")

        return b64_image_str

class ContentBuilder:
    @staticmethod
    def create_text_content(text : str) -> MessageContent:
        return MessageContent(type=ContentType.TEXT, data=text)

    @staticmethod
    def create_image_content(image: Union[Path, np.ndarray]) -> MessageContent:
        b64_image_str = ImageEncoder.encode_image(image)
        return MessageContent(type=ContentType.IMAGE, data=b64_image_str)

class MessageBuilder:
    def __init__(self, role : MessageRole):
        self._role = role
        self._content: List[MessageContent] = []

    def add_text_content(self, text: str):
        self._content.append(ContentBuilder.create_text_content(text))

    def add_image_content(self, image: Union[Path, np.ndarray]):
        self._content.append(ContentBuilder.create_image_content(image))

    def build_message(self) -> Message:
        message = None

        if not self._role:
            raise MessageBuilderError("Message role cannot be empty")

        if not self._content:
            raise MessageBuilderError("Message content cannot be empty")

        if len(self._content) == 1:
            # If there's only one content item, return it directly
            message = Message(role=self._role, content=self._content[0].data)
        else:
            # If there are multiple content items, return them as a list
            message = Message(role=self._role, content=self._content)

        return message