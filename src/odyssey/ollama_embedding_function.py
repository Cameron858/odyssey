from typing import Any, Callable, Dict

import numpy as np
import ollama
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function


@register_embedding_function
class OllamaEmbeddings(EmbeddingFunction):

    def __init__(self, model):
        self.model = model

    def _embed(self, input: Documents) -> ollama.EmbedResponse:
        return ollama.embed(
            model=self.model,
            input=input,
        )

    def __call__(self, input: Documents) -> Embeddings:

        if not input:
            return []

        response = self._embed(input)

        return [np.array(e) for e in response.embeddings]

    @staticmethod
    def name() -> str:
        return "ollama-embeddings"

    def get_config(self) -> Dict[str, Any]:
        return dict(model=self.model)

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction":
        return OllamaEmbeddings(config["model"])
