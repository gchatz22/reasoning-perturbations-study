from openai import OpenAI
from models import LLMBaseModel
from typing import Dict, Any, List
import tiktoken


# number are context windows
OPENAI_MODELS = {
    "gpt-4o": 128000,
    "o1-preview": 128000,
}


class OpenAIModel(LLMBaseModel):
    """
    Implementation of the LLMBaseModel for OpenAI's API.
    """

    def __init__(self, api_key: str, model_name: str, **kwargs):
        super().__init__(api_key, model_name, **kwargs)
        self.client = OpenAI(api_key=self.api_key)

    def generate(
        self, prompt: str, max_tokens: int = 256, temperature: float = 0.7, **kwargs
    ) -> str:
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            model=self.model_name,
            **kwargs
        )
        return response.choices[0].message.content

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
