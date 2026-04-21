"""
tests/test_cortex.py
Unit tests for Cortex AI RAG pipeline.
These run automatically in GitHub Actions CI.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Chunking ──────────────────────────────────────────────────────────────────

def test_chunker_returns_list():
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    docs = [Document(page_content="This is a test. " * 30)]
    chunks = splitter.split_documents(docs)
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_chunks_not_empty():
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = [Document(page_content="Machine learning is a subset of AI. " * 20)]
    chunks = splitter.split_documents(docs)
    for c in chunks:
        assert len(c.page_content.strip()) > 0


# ── Evaluation metrics ────────────────────────────────────────────────────────

def test_faithfulness_range():
    from evaluate_rag_mlflow import faithfulness_score
    score = faithfulness_score(
        "The model uses transformers and attention.",
        ["Transformers and attention mechanisms are used."]
    )
    assert 0.0 <= score <= 1.0


def test_faithfulness_non_negative():
    from evaluate_rag_mlflow import faithfulness_score
    score = faithfulness_score("hello world", ["completely different text here"])
    assert score >= 0.0


def test_answer_relevancy_range():
    from evaluate_rag_mlflow import answer_relevancy_score
    score = answer_relevancy_score(
        "What is RAG?",
        "RAG stands for Retrieval Augmented Generation."
    )
    assert 0.0 <= score <= 1.0


def test_answer_relevancy_unrelated():
    from evaluate_rag_mlflow import answer_relevancy_score
    score = answer_relevancy_score(
        "What is a neural network?",
        "The weather today is sunny and warm outside."
    )
    assert score < 0.5


def test_evaluation_pipeline_keys():
    from evaluate_rag_mlflow import run_evaluation
    from evaluate_rag_mlflow import EVAL_QUESTIONS
    results = run_evaluation(EVAL_QUESTIONS[:2], chain=None)
    assert "avg_faithfulness" in results
    assert "avg_answer_relevancy" in results
    assert "per_question" in results
    assert results["num_questions"] == 2


# ── Imports ───────────────────────────────────────────────────────────────────

def test_langchain_imports():
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    assert True


def test_mlflow_import():
    import mlflow
    mlflow.set_experiment("test-cortex-ci")
    exp = mlflow.get_experiment_by_name("test-cortex-ci")
    assert exp is not None