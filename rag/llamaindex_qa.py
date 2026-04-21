"""
Cortex AI — LlamaIndex PDF Q&A
================================
Add this to your rag/ folder alongside langchain_retriever.py.
Shows you know TWO RAG frameworks — LangChain AND LlamaIndex.

Resume skills line add:
  "LlamaIndex VectorStoreIndex, LangChain RetrievalQA"
"""

import os, sys
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL  = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE   = 512
CHUNK_OVERLAP= 64
TOP_K        = 3
PERSIST_DIR  = str(Path(__file__).parent.parent / "vector_db" / "llamaindex_storage")


# ── Settings ──────────────────────────────────────────────────────────────────

def configure():
    Settings.embed_model   = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.llm           = None          # no LLM — retrieval only demo
    Settings.chunk_size    = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP
    print(f"[LlamaIndex] Embed model: {EMBED_MODEL}")


# ── Build index ───────────────────────────────────────────────────────────────

def build_index(pdf_path: str) -> VectorStoreIndex:
    pdf_dir  = str(Path(pdf_path).parent)
    reader   = SimpleDirectoryReader(input_dir=pdf_dir, required_exts=[".pdf"])
    docs     = reader.load_data()
    print(f"[LlamaIndex] Loaded {len(docs)} doc(s)")

    parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes  = parser.get_nodes_from_documents(docs)
    print(f"[LlamaIndex] Created {len(nodes)} nodes")

    index = VectorStoreIndex(nodes, show_progress=True)
    os.makedirs(PERSIST_DIR, exist_ok=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"[LlamaIndex] Index saved → {PERSIST_DIR}")
    return index


def load_index() -> VectorStoreIndex:
    ctx   = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(ctx)
    print(f"[LlamaIndex] Index loaded from {PERSIST_DIR}")
    return index


# ── Query engine ──────────────────────────────────────────────────────────────

def build_engine(index: VectorStoreIndex):
    retriever = VectorIndexRetriever(index=index, similarity_top_k=TOP_K)
    engine    = RetrieverQueryEngine(retriever=retriever)
    print(f"[LlamaIndex] Query engine ready (top_k={TOP_K})")
    return engine


def query(engine, question: str) -> dict:
    resp    = engine.query(question)
    sources = [
        {"score": round(n.score, 3) if n.score else None,
         "preview": n.text[:80] + "..."}
        for n in resp.source_nodes
    ]
    return {"question": question, "answer": str(resp), "sources": sources}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    configure()

    pdf = sys.argv[1] if len(sys.argv) > 1 else None

    if pdf and os.path.exists(pdf):
        index = build_index(pdf)
    elif os.path.exists(PERSIST_DIR):
        index = load_index()
    else:
        print("[!] Usage: python rag/llamaindex_qa.py path/to/doc.pdf")
        sys.exit(1)

    engine = build_engine(index)

    print("\n[Cortex AI — LlamaIndex] Type 'quit' to exit\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        r = query(engine, q)
        print(f"\nAnswer  : {r['answer']}")
        for s in r["sources"]:
            print(f"  [{s['score']}] {s['preview']}")
        print()
