# WikiRAG

A lightweight, local **Retrieval-Augmented Generation (RAG)** engine that allows you to chat with Wikipedia articles using semantic search.


## Docs

| Doc | Purpose |
|-----|---------|
| **[docs/STATUS.md](./docs/STATUS.md)** | Handoff status |
| [docs/setup.md](./docs/setup.md) | Setup |
| [AGENTS.md](./AGENTS.md) | Agent guidance |

It runs entirely on your CPU after the first model download. The project intentionally uses **encoder-style transformer models**, not a decoder/GPT-style generative LLM.

## Features
- **Local & Private:** Runs 100% offline after model download.
- **CPU Friendly:** Uses PyTorch CPU wheels and relatively small encoder models.
- **Interactive:** Chat with one topic, then switch to another instantly.
- **One-shot CLI:** Query a target article directly from the command line.
- **Smart Chunking:** Uses tokenizer-aware splitting to prevent data loss.
- **Explainable Retrieval:** Optional context output shows which chunks were used.

## Installation

This project uses `uv` for Python version and dependency management. The repo pins Python in `.python-version` and `pyproject.toml`, so `uv` can create the right environment even if your system `python` points at a different version.

1. **Install uv** (if you haven't already):
   ```bash
   pip install uv
   ```
2. **Install the pinned Python version**:
   ```bash
   uv python install 3.11.9
   ```
3. **Create and sync the virtual environment**:
   ```bash
   uv venv --python 3.11.9
   uv sync
   ```

## Usage

### One-shot mode

Ask one question against one Wikipedia article:

```bash
uv run python main.py --topic "Black hole" --question "What is the event horizon?"
```

Show the retrieved chunks that were passed to the QA model:

```bash
uv run python main.py --topic "Black hole" --question "What is the event horizon?" --show-context
```

### Interactive mode

Run the script without arguments:

```bash
uv run python main.py
```

1. Enter a Wikipedia topic (e.g., Napoleon, Black Hole, Linux).

2. Wait for the system to index the text.

3. Ask questions in natural language!

## Architecture

This project is a **retrieval + extractive question-answering** pipeline. It does not call a generative LLM.

Ingestion: `wikipedia` API

Vector search: FAISS (`IndexFlatIP`)

Embedding model: `sentence-transformers/all-mpnet-base-v2`

QA model: `deepset/roberta-base-squad2`

### Pipeline

How it works:

1. Fetches the target Wikipedia article by topic string.
2. Splits the article into overlapping tokenizer-aware chunks.
3. Embeds each chunk with `all-mpnet-base-v2`.
4. Embeds your question with the same embedding model.
5. Retrieves the most similar chunks from FAISS.
6. Passes the question and retrieved text into `deepset/roberta-base-squad2`.
7. Returns a span extracted from the retrieved Wikipedia text.

### Models Used

This project uses two neural models:

| Role | Model | Model family | Purpose |
| --- | --- | --- | --- |
| Embedding encoder | `sentence-transformers/all-mpnet-base-v2` | Transformer encoder | Converts article chunks and questions into vectors for semantic search. |
| QA model | `deepset/roberta-base-squad2` | RoBERTa/BERT-style transformer encoder | Extracts the most likely answer span from retrieved text. |

FAISS is not a language model. It is the vector search layer. It stores the chunk embeddings and returns the nearest chunks for a question embedding.

### Encoder vs Decoder Models

This repo is useful for understanding the difference between **encoder-based retrieval/QA** and **decoder-based generation**.

Encoder models, such as BERT, RoBERTa, MPNet, and many sentence-transformer models, read the input text as a whole and produce representations or labels over that input. They are strong for:

- embeddings
- semantic search
- classification
- reranking
- extractive question answering
- finding answer spans inside known context

Decoder models, such as GPT-style models, generate text token by token. They are strong for:

- open-ended generation
- synthesis across multiple pieces of context
- conversational answers
- summarization
- instruction following
- code and reasoning-style workflows

`deepset/roberta-base-squad2` is a transformer language model, but it is not a GPT-style LLM. It is a RoBERTa encoder fine-tuned on SQuAD-style extractive QA. Given a question and a context window, it predicts the start and end positions of the answer inside that context.

Example:

```text
Question: When was Albert Einstein born?
Context: Albert Einstein was born in Ulm, in the Kingdom of Wurttemberg, on 14 March 1879...
Extractive answer: 14 March 1879
```

A GPT-style generative RAG system would usually retrieve context and then ask a decoder LLM to write a new answer. This project instead retrieves context and asks RoBERTa to select a span from that context. That makes it smaller, local, and easier to inspect, but less flexible than a full generative RAG chatbot.

Current interface: standalone CLI. The `WikiRAGEngine` class can be imported into another Python project, but this repo is not yet packaged as an API service.

## Further Work

Good next steps for this project:

1. **Better CLI ergonomics**
   - Add `--json` output for scripts.
   - Add `--top-k`, `--chunk-size`, and `--chunk-overlap` flags.
   - Add `--page-title` output so users can see exactly which Wikipedia page was selected.

2. **Index caching**
   - Cache article text and FAISS indexes by topic.
   - Avoid re-downloading and re-embedding the same article on every run.
   - Store metadata such as page title, URL, model name, and chunk settings.

3. **Source citations**
   - Return the retrieved chunk text and article URL with every answer.
   - Track section headings from the Wikipedia article.
   - Show answer confidence plus retrieval scores.

4. **Improved retrieval quality**
   - Add a cross-encoder reranker after FAISS retrieval.
   - Try smaller/faster embedding models for CPU usage.
   - Compare cosine similarity, L2 distance, and normalized inner product.
   - Tune chunk size and overlap.

5. **Generative RAG mode**
   - Add an optional decoder model such as a local instruction-tuned model.
   - Keep the current extractive mode as the lightweight baseline.
   - Compare extractive answers vs generated answers over the same retrieved chunks.

6. **Multi-article retrieval**
   - Search and index multiple related Wikipedia pages.
   - Let users pass several topics.
   - Add a discovery step that follows links from the seed article.

7. **API service**
   - Wrap `WikiRAGEngine` with FastAPI.
   - Add endpoints for indexing, querying, and cache management.
   - Keep the CLI as a thin client over the same engine.

8. **Evaluation**
   - Add a small benchmark set of topics/questions/expected answers.
   - Measure retrieval hit rate and QA answer quality.
   - Track latency for model loading, indexing, and querying.

9. **Packaging**
   - Move engine code into a package directory such as `wikirag/`.
   - Expose a console script entry point.
   - Separate CLI parsing from core retrieval/QA logic.

## Tests

Run the lightweight unit tests with:

```bash
uv run python -m unittest discover -s tests
```
