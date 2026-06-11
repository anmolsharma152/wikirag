import contextlib
import io
import types
import unittest

import numpy as np

import main


class FakeTokenizer:
    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)


class FakeEmbedder:
    def encode(self, values, show_progress_bar=False):
        if isinstance(values, str):
            values = [values]
        vectors = []
        for index, _ in enumerate(values):
            vectors.append([float(index + 1), 0.0])
        return np.asarray(vectors, dtype="float32")


class FakeIndex:
    def __init__(self, dimension):
        self.dimension = dimension
        self.vectors = None
        self.last_k = None

    def add(self, vectors):
        self.vectors = vectors

    def search(self, query_embedding, k):
        self.last_k = k
        scores = self.vectors @ query_embedding[0]
        indices = np.argsort(scores)[::-1][:k]
        return scores[indices].reshape(1, -1), indices.reshape(1, -1)


class WikiRAGEngineTests(unittest.TestCase):
    def make_engine(self):
        engine = main.WikiRAGEngine.__new__(main.WikiRAGEngine)
        engine.embedder = FakeEmbedder()
        engine.tokenizer = FakeTokenizer()
        engine.qa_pipeline = lambda question, context: {
            "answer": context,
            "score": 0.75,
        }
        engine.chunks = []
        engine.index = None
        engine.current_topic = None
        return engine

    def build_index_quietly(self, engine, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return engine.build_index(*args, **kwargs)

    def test_query_without_index_returns_structured_error(self):
        result = self.make_engine().query("What happened?")

        self.assertEqual("", result["answer"])
        self.assertEqual(0.0, result["score"])
        self.assertEqual([], result["context_used"])
        self.assertIn("error", result)

    def test_build_index_rejects_overlap_that_would_not_advance(self):
        engine = self.make_engine()

        with self.assertRaises(ValueError):
            engine.build_index("one two three", chunk_size=2, chunk_overlap=2)

    def test_build_index_uses_normalized_vectors(self):
        engine = self.make_engine()
        original_faiss = main.faiss
        main.faiss = types.SimpleNamespace(IndexFlatIP=FakeIndex)
        try:
            self.assertTrue(self.build_index_quietly(
                engine,
                "one two three",
                chunk_size=2,
                chunk_overlap=1,
            ))
        finally:
            main.faiss = original_faiss

        norms = np.linalg.norm(engine.index.vectors, axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms))

    def test_query_clamps_k_to_available_chunks(self):
        engine = self.make_engine()
        engine.chunks = ["only chunk"]
        engine.index = FakeIndex(2)
        engine.index.add(np.asarray([[1.0, 0.0]], dtype="float32"))

        result = engine.query("question", k=3)

        self.assertEqual(1, engine.index.last_k)
        self.assertEqual(["only chunk"], result["context_used"])
        self.assertEqual("only chunk", result["answer"])


class CliTests(unittest.TestCase):
    def test_print_result_can_include_context(self):
        result = {
            "answer": "Ada Lovelace",
            "score": 0.91,
            "context_used": ["Ada wrote notes.", "The notes described an algorithm."],
        }

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main.print_result(result, show_context=True)

        text = output.getvalue()
        self.assertIn("Answer: Ada Lovelace", text)
        self.assertIn("Confidence: 0.91", text)
        self.assertIn("[1] Ada wrote notes.", text)

    def test_main_requires_topic_and_question_together(self):
        original_parse_args = main.parse_args
        main.parse_args = lambda: types.SimpleNamespace(
            topic="Ada Lovelace",
            question=None,
            k=3,
            show_context=False,
        )
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(2, main.main())
        finally:
            main.parse_args = original_parse_args


if __name__ == "__main__":
    unittest.main()
