from typing import Any, Literal, TypedDict

from ollama import Client


class ChatMessage(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


class SimpleChat:
    def __init__(self, model: str) -> None:
        self.model = model
        self._client = Client()

    def __call__(self, message: str, history: list[ChatMessage]) -> Any:
        return self._chat(message, history)

    def _chat(self, message: str, history: list[ChatMessage]) -> str:
        response = self._client.chat(
            model=self.model,
            messages=[*history, {"role": "user", "content": message}],
        )

        if response.message.content is None:
            raise RuntimeError("Ollama returned no content")

        return response.message.content


if __name__ == "__main__":

    chat = SimpleChat("llama3.1:8b")

    response = chat("Hi there.", [])

    print(response)
