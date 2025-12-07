"""
Setup configuration for NLQ library
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nlqcat",
    version="0.1.0",
    author="Anirban-QuantumCAT",
    author_email="sanirban2006@gmail.com",
    url="https://github.com/Anirbansarkars/nlqcat",
    description="NLQCAT - Natural Language Query Framework (Hybrid NLP + AI + VectorDB)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    packages=find_packages(include=["nlqcat", "nlqcat.*"]),
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],

    python_requires=">=3.8",

    install_requires=[
        "spacy>=3.0.0",
        "scikit-learn>=1.0.0",
        "sentence-transformers>=2.2.0",
        "chromadb>=0.5.0",
        "faiss-cpu>=1.7.4",
        "openai>=1.0.0",
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "accelerate>=0.20.0",
        "ctransformers>=0.2.0; python_version < '3.11'",   # works only < 3.11
        "numpy>=1.18.0",
    ],

    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },

    include_package_data=True,
)
