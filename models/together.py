from together import Together
from models import LLMBaseModel
from typing import Dict, Any, List
import tiktoken


# numbers are context windows
TOGETHERAI_MODELS = {
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": 128000,
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": 128000,
    "meta-llama/Llama-3.2-3B-Instruct-Turbo": 128000,
    "google/gemma-2-9b-it": 8192,
    "google/gemma-2-27b-it": 200000,
    "mistralai/Mistral-7B-Instruct-v0.3": 32768,
    "mistralai/Mixtral-8x22B-Instruct-v0.1": 64000,
}


class TogetherAIModel(LLMBaseModel):
    """
    Implementation of the LLMBaseModel for TogetherAI's API.
    """

    def __init__(self, api_key: str, model_name: str, **kwargs):
        super().__init__(api_key, model_name, **kwargs)
        self.client = Together(api_key=api_key)

    def generate(
        self, prompt: str, max_tokens: int = 1000, temperature: float = 0.2, **kwargs
    ) -> str:
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            # max_tokens=max_tokens,
            model=self.model_name,
            **kwargs
        )
        return response.choices[0].message.content

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "provider": "TogetherAI",
            "model_name": self.model_name,
        }

    def set_parameters(self, **kwargs) -> None:
        self.config.update(kwargs)

    def tokenize(self, text) -> List[int]:
        # NOTE This is incorrect but it is relatively tedious to use a different tokenizer for each model. This serves as a rough estimate.
        encoding = tiktoken.encoding_for_model("gpt-4o")
        return encoding.encode(text)

    def detokenize(self, token_ids) -> str:
        # NOTE This is incorrect but it is relatively tedious to use a different tokenizer for each model. This serves as a rough estimate.
        encoding = tiktoken.encoding_for_model("gpt-4o")
        return encoding.decode(token_ids)

    def max_token_limit(self) -> int:
        return TOGETHERAI_MODELS[self.model_name]
