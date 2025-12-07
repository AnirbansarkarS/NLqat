from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class LLMBase(ABC):
    """
    Abstract Base Class for LLM Integration.
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
