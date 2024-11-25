import anthropic
from models import LLMBaseModel
from typing import Dict, Any, List


# number are context windows
ANTHROPIC_MODELS = {
    "claude-3-5-sonnet-latest": 200000,
}


class AnthropicModel(LLMBaseModel):
    """
    Implementation of the LLMBaseModel for Anthropic's API.
    """

    def __init__(self, api_key: str, model_name: str, **kwargs):
        super().__init__(api_key, model_name, **kwargs)
        self.client = anthropic.Client(api_key=api_key)

    def generate(
        self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7, **kwargs
    ) -> str:
        response = self.client.messages.create(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            model=self.model_name,
            **kwargs
        )
        return response.content[0].text

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "provider": "Anthropic",
            "model_name": self.model_name,
        }

    def set_parameters(self, **kwargs) -> None:
        self.config.update(kwargs)

    def tokenize(self, text) -> List[int]:
        return []

    def detokenize(self, token_ids) -> str:
        return ""

    def max_token_limit(self) -> int:
        return ANTHROPIC_MODELS[self.model_name]
