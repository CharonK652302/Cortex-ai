"""
Cortex AI — LangChain RAG Pipeline (LCEL style)
=================================================
Uses LangChain Expression Language (LCEL) — the modern
approach that works with langchain >= 0.2.
No langchain.chains dependency needed.

Resume bullet:
  "Built production RAG pipeline using LangChain LCEL +
   FAISS vector store + sentence-transformer embeddings"
"""

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline as hf_pipeline

EMBED_MODEL   = "all-mpnet-base-v2"
LLM_MODEL     = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
TOP_K         = 3
INDEX_PATH    = str(Path(__file__).parent.parent / "vector_db" / "langchain_index")

RAG_PROMPT = PromptTemplate.from_template("""
Use the following context to answer the question.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question: {question}

Answer:""")


def load_and_chunk(pdf_path: str):
    loader   = PyPDFLoader(pdf_path)
    docs     = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"[LangChain] Loaded {len(docs)} pages -> {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks):
    embeddings  = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(INDEX_PATH, exist_ok=True)
    vectorstore.save_local(INDEX_PATH)
    print(f"[LangChain] FAISS index saved -> {INDEX_PATH}")
    return vectorstore


def load_vectorstore():
    embeddings  = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.load_local(
        INDEX_PATH, embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"[LangChain] FAISS index loaded from {INDEX_PATH}")
    return vectorstore


def build_chain(vectorstore):
    pipe = hf_pipeline(
        "text-generation",
        model=LLM_MODEL,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=True,
        pad_token_id=2,
    )
    llm       = HuggingFacePipeline(pipeline=pipe)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    print(f"[LangChain] LCEL RAG chain ready (top_k={TOP_K})")
    return chain, retriever


def query_chain(chain_and_retriever, question: str) -> dict:
    chain, retriever = chain_and_retriever
    answer  = chain.invoke(question)
    sources = retriever.invoke(question)
    pages   = [f"page {d.metadata.get('page','?')}" for d in sources]
    return {
        "question"        : question,
        "answer"          : answer,
        "source_pages"    : pages,
        "chunks_retrieved": len(sources),
    }


if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else None

    if pdf and os.path.exists(pdf):
        print(f"\n[*] Indexing: {pdf}")
        chunks      = load_and_chunk(pdf)
        vectorstore = build_vectorstore(chunks)
    elif os.path.exists(INDEX_PATH):
        print(f"\n[*] Loading existing index...")
        vectorstore = load_vectorstore()
    else:
        print("[OK] Imports successful!")
        print("[!]  To run with PDF: python rag/langchain_retriever.py path/to/doc.pdf")
        raise SystemExit(0)

    chain_and_retriever = build_chain(vectorstore)
    print("\n[Cortex AI - LangChain LCEL RAG] Type 'quit' to exit\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        r = query_chain(chain_and_retriever, q)
        print(f"\nAnswer  : {r['answer']}")
        print(f"Sources : {', '.join(r['source_pages'])}")
        print(f"Chunks  : {r['chunks_retrieved']}\n")