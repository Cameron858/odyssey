def gradio_chat_adapter(model):
    def chat(message, history):
        ollama_history = [
            {
                "role": item["role"],
                "content": item["content"][0]["text"],
            }
            for item in history
        ]

        return model(message, ollama_history)

    return chat
