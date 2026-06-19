from typing import Any, Dict, List

import numpy as np
import ollama
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.api.types import Space
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

        return [
            np.array(embedding, dtype=np.float32) for embedding in response.embeddings
        ]

    @staticmethod
    def name() -> str:
        return "ollama-embeddings"

    def default_space(self) -> Space:
        return "cosine"

    def supported_spaces(self) -> List[Space]:
        return ["cosine", "l2", "ip"]

    def get_config(self) -> Dict[str, Any]:
        return dict(model=self.model)

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction":
        return OllamaEmbeddings(config["model"])
