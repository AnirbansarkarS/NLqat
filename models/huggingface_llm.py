from .llm_base import LLMBase
from typing import List, Dict, Any

class HuggingFaceLLM(LLMBase):
    def __init__(self, model_name_or_path: str = "gpt2", use_gpu: bool = False):
        from transformers import pipeline
        import torch
        
        device = 0 if use_gpu and torch.cuda.is_available() else -1
        print(f"Loading HuggingFace model: {model_name_or_path} on device {device}...")
        
        self.pipeline = pipeline(
            "text-generation",
            model=model_name_or_path,
            device=device,
            max_new_tokens=256
        )

    def generate(self, prompt: str, max_tokens: int = 100, **kwargs) -> str:
        # Update max_length or max_new_tokens based on args
        # Transformers pipeline returns a list of dicts
        output = self.pipeline(prompt, max_new_tokens=max_tokens, **kwargs)
        # return generated text. Output[0]['generated_text'] usually includes prompt.
        generated_text = output[0]['generated_text']
        # Simple heuristic to strip prompt if desired, but standard generation usually includes it.
        # We'll return full text for now as is common with completion models.
        return generated_text

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # Basic chat template implementation if model supports it, 
        # otherwise just concat messages.
        # For simplicity in this phase, we concat last user message.
        last_msg = messages[-1]['content']
        return self.generate(last_msg, **kwargs)
