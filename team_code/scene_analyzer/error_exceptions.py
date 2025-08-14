class ImageEncoderError(Exception):
    """Custom exception for errors related to image encoding."""
    def __init__(self, message: str):
        super().__init__(message)

class MessageBuilderError(Exception):
    """Custom exception for errors related to message building."""
    def __init__(self, message: str):
        super().__init__(message)

class APIClientError(Exception):
    """Custom exception for errors related to OpenRouter client operations."""
    def __init__(self, message: str):
        super().__init__(message)