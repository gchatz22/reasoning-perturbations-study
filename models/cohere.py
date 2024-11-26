import cohere
from models import LLMBaseModel
from typing import Dict, Any, List
import tiktoken


# numbers are context windows
COHERE_MODELS = {
    "command-r-plus-08-2024": 128000,
}


class CohereModel(LLMBaseModel):
    """
    Implementation of the LLMBaseModel for Cohere's API.
    """

    def __init__(self, api_key: str, model_name: str, **kwargs):
        super().__init__(api_key, model_name, **kwargs)
        self.client = cohere.Client(api_key=api_key)

    def generate(
        self, prompt: str, max_tokens: int = 2000, temperature: float = 0.2, **kwargs
    ) -> str:
        response = self.client.chat(
            message=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=self.model_name,
            **kwargs
        )
        return response.text

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "provider": "Cohere",
            "model_name": self.model_name,
        }

    def set_parameters(self, **kwargs) -> None:
        self.config.update(kwargs)

    def tokenize(self, text) -> List[int]:
        response = self.client.tokenize(text=text, model=self.model_name)
        return response.tokens

    def detokenize(self, token_ids) -> str:
        response = self.client.detokenize(tokens=token_ids, model=self.model_name)
        return response.text

    def max_token_limit(self) -> int:
        return COHERE_MODELS[self.model_name]
