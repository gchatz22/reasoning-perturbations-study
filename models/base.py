from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union


class LLMBaseModel(ABC):
    """
    Abstract base class for a Large Language Model (LLM) API interface.
    Subclasses must implement methods to interact with specific LLM providers.
    """

    def __init__(self, api_key: str, model_name: str, **kwargs):
        """
        Initialize the base model with API key and model name.
        Additional provider-specific parameters can be passed via kwargs.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.2, **kwargs) -> str:
        """
        Generate text from the model given a prompt.
        Subclasses must implement this method.
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Retrieve metadata about the model, such as version or provider info.
        Subclasses must implement this method.
        """
        pass

    @abstractmethod
    def set_parameters(self, **kwargs) -> None:
        """
        Set runtime parameters for the model.
        Subclasses must implement this method.
        """
        pass

    @abstractmethod
    def tokenize(self, text) -> List[int]:
        """
        Tokenize the given text for the given model and return the list of token ids.
        """
        pass

    @abstractmethod
    def detokenize(self, token_ids) -> str:
        """
        Detokenize the given token ids for the given model and return the text.
        """
        pass

    @abstractmethod
    def max_token_limit(self) -> int:
        """
        Returns the context window for the given model
        """
        pass
