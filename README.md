# NLQcat: Natural Language Question Answering Toolkit

**NLQcat** is a production-ready, hybrid NLP + GenAI library that unifies classic linguistic analysis (spaCy) with modern semantic search (Vector Databases) and Large Language Models (LLMs) into a single, easy-to-use pipeline.

## 📦 Installation

```bash
pip install nlqcat
python -m spacy download en_core_web_sm
```

## 🚀 Quick Start (3-Line Example)

```python
from nlqcat.core.pipeline import Pipeline

pipe = Pipeline(vector_store_type="chroma")
pipe.add_documents(["NLQcat combines linguistic NLP with semantic RAG."])
print(pipe.query("What does NLQcat do?")['answer'])
```

## ✨ Features

- **Hybrid Analysis**: Seamlessly mixes spaCy's linguistic features (NER, POS tagging) with semantic embeddings.
- **Unified RAG Pipeline**: Built-in Retrieve-Augmented Generation flow (Query -> Retrieve -> Answer).
- **Vector Store Support**: Integrated ChromaDB client for local vector storage.
- **Modular Design**: Plug-and-play components for Loaders, Preprocessors, and Agents.
- **LLM Ready**: Simple `OpenAIAgent` integration for generative answers.

## 🗺️ Roadmap

- [ ] Support for FAISS and Pinecone vector stores.
- [ ] Integration with HuggingFace and Anthropic LLMs.
- [ ] Advanced RAG techniques (Re-ranking, HyDE).
- [ ] Async support for high-throughput pipelines.