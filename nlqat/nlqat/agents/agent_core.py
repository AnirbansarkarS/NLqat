from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os

class LLMBase(ABC):
    """
    Abstract Base Class for Agent/LLM Integration.
    """
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 100, **kwargs) -> str:
        """
        Generates text based on a prompt.
        """
        pass

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Chat completion interface.
        messages: List of dicts with 'role' and 'content' keys.
        """
        pass

class OpenAIAgent(LLMBase):
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        try:
            import openai
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
            
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 100, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            **kwargs
        )
        return response.choices[0].message.content.strip()

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content.strip()
