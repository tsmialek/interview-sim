import json

import ollama

from log_runs import SlmConfig
from process import SLM_SCHEMA, SLM_SYSTEM_PROMPT
from utils import get_sha256


class OllamaSlmClient:
    def __init__(self, config: SlmConfig):
        self.config = config

    def chat(self, system: str, user: str, schema: dict) -> str:
        response = ollama.chat(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": user,
                },
            ],
            format=schema,
            options={"temperature": self.config.temperature, "num_predict": 1000},
        )
        if response.message.content is None:
            return ""

        return response.message.content


def make_slm_config(model: str, temperature: float) -> SlmConfig:
    config_json_string = json.dumps(SLM_SCHEMA, sort_keys=True)

    prompt_sha = get_sha256(config_json_string + SLM_SYSTEM_PROMPT, length=8)

    digest = None
    for m in ollama.list().models:
        if m.model == model:
            digest = m.digest
    if digest is None:
        raise ValueError("Model is not downloaded in Ollama")

    return SlmConfig(
        model=model, digest=digest, prompt_sha=prompt_sha, temperature=temperature
    )
