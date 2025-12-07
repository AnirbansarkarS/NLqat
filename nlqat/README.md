# NLqat

NLqat is a hybrid NLP and GenAI toolkit that combines classic linguistic analysis (spaCy) with modern semantic search (vector stores, embeddings) and LLM reasoning.

## Installation

```bash
pip install nlqat
```

## Usage

```python
from nlqat.pipeline import Pipeline
# from nlqat.agents import OpenAIAgent

pipe = Pipeline(enable_spacy=True, vector_store_type="chroma")
pipe.add_documents(["Doc 1 content", "Doc 2 content"])
result = pipe.query("What is in Doc 1?")
print(result['answer'])
```
