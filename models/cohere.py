import cohere
from models import LLMBaseModel
from typing import Dict, Any, List
import tiktoken


# numbers are context windows
COHERE_MODELS = {
    "command-r-08-24": 128000,
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
            model="command",
            **kwargs
        )
        return response.text

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "provider": "OpenAI",
            "model_name": self.model_name,
        }

    def set_parameters(self, **kwargs) -> None:
        self.config.update(kwargs)

    def tokenize(self, text) -> List[int]:
        encoding = tiktoken.encoding_for_model(self.model_name)
        return encoding.encode(text)

    def detokenize(self, token_ids) -> str:
        encoding = tiktoken.encoding_for_model(self.model_name)
        return encoding.decode(token_ids)

    def max_token_limit(self) -> int:
        return OPENAI_MODELS[self.model_name]
