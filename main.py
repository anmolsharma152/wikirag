import argparse
import faiss
import numpy as np
import sys
import wikipedia
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline


def _to_float32_matrix(values):
    array = np.asarray(values, dtype="float32")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array


def _normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class WikiRAGEngine:
    def __init__(self,
                 embedding_model="sentence-transformers/all-mpnet-base-v2",
                 qa_model="deepset/roberta-base-squad2",
                 device=-1): # device=-1 for CPU, 0 for GPU

        print(f"Loading models... (This may take a moment)")

        # 1. Load Embedding Model (for retrieval)
        try:
            self.embedder = SentenceTransformer(embedding_model)
            # We load the tokenizer explicitly for accurate chunking
            self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            sys.exit(1)

        # 2. Load QA Model (for answering)
        try:
            self.qa_pipeline = pipeline(
                "question-answering",
                model=qa_model,
                tokenizer=qa_model,
                device=device
            )
        except Exception as e:
            print(f"Error loading QA model: {e}")
            sys.exit(1)

        # Internal storage
        self.chunks = []
        self.index = None
        self.current_topic = None

    def load_content(self, topic):
        """Fetches content from Wikipedia with error handling."""
        print(f"\nSearching Wikipedia for: '{topic}'...")
        try:
            page = wikipedia.page(topic)
            self.current_topic = page.title
            print(f"Successfully loaded: {page.title}")
            return page.content
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"\n⚠️ Ambiguous topic. Did you mean one of these?")
            # Show first 5 options
            for option in e.options[:5]:
                print(f" - {option}")
            return None
        except wikipedia.exceptions.PageError:
            print(f"\n❌ Page not found for '{topic}'. Try a different keyword.")
            return None
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {e}")
            return None

    def build_index(self, text, chunk_size=256, chunk_overlap=20):
        """Chunks text and builds the FAISS index."""
        if not text:
            return False
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        print("Processing text and building index...")

        # 1. Accurate Chunking using the Tokenizer
        tokens = self.tokenizer.tokenize(text)
        self.chunks = []

        # Sliding window approach
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_text = self.tokenizer.convert_tokens_to_string(tokens[start:end])
            self.chunks.append(chunk_text)

            if end == len(tokens):
                break
            start = end - chunk_overlap

        print(f"Created {len(self.chunks)} text chunks.")

        # 2. Create Embeddings
        embeddings = _normalize_rows(_to_float32_matrix(
            self.embedder.encode(self.chunks, show_progress_bar=True)
        ))

        # 3. Build FAISS Index. Inner product over normalized vectors is cosine similarity.
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        return True

    def query(self, question, k=3):
        """Retrieves context and generates an answer."""
        if self.index is None or not self.chunks:
            return {
                "answer": "",
                "score": 0.0,
                "context_used": [],
                "error": "Index not built. Please load a topic first."
            }
        if k <= 0:
            raise ValueError("k must be greater than 0")

        # 1. Retrieve relevant chunks
        k = min(k, len(self.chunks))
        query_embedding = _normalize_rows(_to_float32_matrix(
            self.embedder.encode([question])
        ))
        distances, indices = self.index.search(np.array(query_embedding), k)

        retrieved_chunks = [
            self.chunks[i]
            for i in indices[0]
            if 0 <= i < len(self.chunks)
        ]
        if not retrieved_chunks:
            return {
                "answer": "",
                "score": 0.0,
                "context_used": [],
                "error": "No relevant context found."
            }
        context = " ".join(retrieved_chunks)

        # 2. Generate Answer
        result = self.qa_pipeline(question=question, context=context)

        return {
            "answer": result['answer'],
            "score": result['score'],
            "context_used": retrieved_chunks
        }


def print_result(result, show_context=False):
    if result.get("error"):
        print(f"\nError: {result['error']}")
        return

    print(f"\nAnswer: {result['answer']}")
    print(f"Confidence: {result['score']:.2f}")
    if show_context:
        print("\nContext used:")
        for index, chunk in enumerate(result["context_used"], start=1):
            print(f"\n[{index}] {chunk}")


def run_one_shot(topic, question, k=3, show_context=False):
    rag = WikiRAGEngine()
    content = rag.load_content(topic)
    if not content or not rag.build_index(content):
        return 1

    result = rag.query(question, k=k)
    print_result(result, show_context=show_context)
    return 1 if result.get("error") else 0


def run_interactive():
    rag = WikiRAGEngine()

    while True:
        print("\n" + "="*40)
        topic = input("Enter a Wikipedia topic (or 'exit' to quit'): ").strip()

        if topic.lower() == 'exit':
            break
        if not topic:
            continue

        content = rag.load_content(topic)

        if content:
            success = rag.build_index(content)
            if success:
                print("\n✅ System Ready! Ask questions about the topic.")
                print("Type 'new' to change topic or 'exit' to quit.")

                while True:
                    question = input("\nYour Question: ").strip()

                    if question.lower() == 'exit':
                        sys.exit(0)
                    if question.lower() == 'new':
                        break # Break inner loop to choose new topic
                    if not question:
                        continue

                    result = rag.query(question)

                    print_result(result)
                    # Optional: Print source context
                    # print(f"   Context: {result['context_used'][0][:100]}...")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ask questions over a Wikipedia article with local retrieval and extractive QA."
    )
    parser.add_argument("--topic", help="Wikipedia topic/page to retrieve and index.")
    parser.add_argument("--question", help="Question to answer against the selected Wikipedia article.")
    parser.add_argument("--k", type=int, default=3, help="Number of retrieved chunks to use as context.")
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the retrieved chunks used by the QA model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.topic or args.question:
        if not args.topic or not args.question:
            print("Both --topic and --question are required for non-interactive mode.", file=sys.stderr)
            return 2
        return run_one_shot(
            topic=args.topic,
            question=args.question,
            k=args.k,
            show_context=args.show_context,
        )

    run_interactive()
    return 0


if __name__ == "__main__":
    sys.exit(main())
