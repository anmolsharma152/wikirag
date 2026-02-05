# WikiRAG 🧠

A lightweight, local **Retrieval-Augmented Generation (RAG)** engine that allows you to chat with Wikipedia articles using semantic search.

It runs entirely on your CPU using `sentence-transformers` for embeddings and `RoBERTa` for extractive question answering.

## 🚀 Features
- **Local & Private:** Runs 100% offline after model download.
- **CPU Optimized:** Uses quantized/lightweight versions of PyTorch.
- **Interactive:** Chat with one topic, then switch to another instantly.
- **Smart Chunking:** Uses tokenizer-aware splitting to prevent data loss.

## 🛠️ Installation

This project uses `uv` for lightning-fast dependency management.

1. **Install uv** (if you haven't already):
   ```bash
   pip install uv
   ```
2. Set up the environment:
   ```bash
   uv venv
   source .venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   # Install lightweight PyTorch first
   uv pip install torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
   # Install the rest
   uv pip install sentence-transformers faiss-cpu wikipedia transformers
   ```

## 💡 Usage
 **Simply run the main script**:
 
   ```bash
   python main.py
   ```
1. Enter a Wikipedia topic (e.g., Napoleon, Black Hole, Linux).

2. Wait for the system to index the text.

3. Ask questions in natural language!

## 🏗️ Architecture
   
Ingestion: wikipedia API

Vector Database: faiss (Facebook AI Similarity Search)

Embeddings: all-mpnet-base-v2

QA Model: deepset/roberta-base-squad2

