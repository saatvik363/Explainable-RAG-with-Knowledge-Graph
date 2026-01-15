# Explainable-RAG-with-Knowledge-Graph

This repository contains a local Retrieval-Augmented Generation (RAG) system built for the **DataForge Challenge**. It retrieves information from local PDF documents, generates a natural language answer using **LLaMA-3**, and provides an explainability layer via entity co-occurrence mapping.

## 🏗 System Architecture

The project consists of three modular components orchestrated by `main.py`:

1.  **Ingestion & Retrieval (`retriever.py`)**:
    * Loads PDFs using `PyPDFParser`.
    * Splits text into 500-character chunks (50 overlap).
    * Embeds chunks using `BAAI/bge-small-en` (HuggingFace).
    * Stores vectors locally using **FAISS**.
2.  **Explainability Layer (`npl_processor.py`)**:
    * Filters retrieved chunks based on semantic similarity to the query.
    * Uses **spaCy** (`en_core_web_md`) to extract Named Entities.
    * Generates a co-occurrence map to visualize relationships between entities in the retrieved context.
3.  **Generative Answer (`answer_generator.py`)**:
    * Constructs a strict JSON-prompted context.
    * Queries a local **Ollama** instance (LLaMA-3-8b).
    * Returns the answer with cited chunk IDs for verification.

## 🛠 Prerequisites

Ensure you have the following installed before running the project:

* **Python 3.10+**
* **Ollama:** Must be installed and running locally. [Download Ollama](https://ollama.com/)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-folder>
