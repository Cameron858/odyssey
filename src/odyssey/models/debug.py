class DebugChat:

    def __call__(self, message: str, history: list[dict]) -> str:
        return f"{message=}\n{history=}"
