import wikipedia
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
import sys

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
        embeddings = self.embedder.encode(self.chunks, show_progress_bar=True)

        # 3. Build FAISS Index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings))

        return True

    def query(self, question, k=3):
        """Retrieves context and generates an answer."""
        if not self.index:
            return "Index not built. Please load a topic first."

        # 1. Retrieve relevant chunks
        query_embedding = self.embedder.encode([question])
        distances, indices = self.index.search(np.array(query_embedding), k)

        retrieved_chunks = [self.chunks[i] for i in indices[0]]
        context = " ".join(retrieved_chunks)

        # 2. Generate Answer
        result = self.qa_pipeline(question=question, context=context)

        return {
            "answer": result['answer'],
            "score": result['score'],
            "context_used": retrieved_chunks
        }

def main():
    # Initialize the engine once
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

                    print(f"\n🔎 Answer: {result['answer']}")
                    print(f"   (Confidence: {result['score']:.2f})")
                    # Optional: Print source context
                    # print(f"   Context: {result['context_used'][0][:100]}...")

if __name__ == "__main__":
    main()
