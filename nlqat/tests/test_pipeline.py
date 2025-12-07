import shutil
import os
import sys
import unittest

# Ensure we can import nlqat. Adjust path if running from tests dir directly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nlqat.pipeline import Pipeline
from nlqat.agents.agent_core import LLMBase

class MockLLM(LLMBase):
    def generate(self, prompt: str, max_tokens: int = 100, **kwargs) -> str:
        return f"Mock answer for: {prompt[:20]}..."

    def chat(self, messages, **kwargs) -> str:
        return "Mock chat response"

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.test_db_path = "./pipeline_test_db"
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)

    def tearDown(self):
        if os.path.exists(self.test_db_path):
            try:
                shutil.rmtree(self.test_db_path)
            except:
                pass

    def test_pipeline_flow(self):
        print("Initialize Unified Pipeline...")
        
        pipe = Pipeline(
            enable_spacy=True,
            vector_store_type="chroma",
            vector_store_path=self.test_db_path,
            llm=MockLLM()
        )
        
        docs = [
            "The project is named HybridAI.",
            "It combines spaCy and semantic transformers.",
            "Phase 5 is the unified pipeline engine."
        ]
        
        print("\n--- Adding Documents ---")
        pipe.add_documents(docs)
        
        print("\n--- Querying Pipeline ---")
        question = "What is the project name and what does it combine?"
        result = pipe.query(question)
        
        # Verify Analysis
        self.assertTrue(result['analysis']['entities'] is not None)
        
        # Verify Retrieval
        self.assertTrue(len(result['retrieved_docs']) > 0)
        retrieved_texts = [d['text'] for d in result['retrieved_docs']]
        self.assertTrue(any("HybridAI" in t for t in retrieved_texts))
        
        # Verify Answer
        self.assertTrue("Mock answer" in result['answer'])
        print("\nSUCCESS: Pipeline verified.")

if __name__ == "__main__":
    unittest.main()
