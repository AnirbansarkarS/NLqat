class MockLLM:
    def generate_response(self, prompt: str) -> str:
        return f"[MOCK RESPOSNE] I received the prompt: {prompt[:50]}... and I am answering it."
