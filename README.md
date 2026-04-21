# 🧠 Cortex AI — Production RAG App

> Chat with any PDF using AI. Built with LangChain LCEL, LlamaIndex, Crew.ai multi-agent pipeline, FAISS, FastAPI, Docker, MLflow evaluation, GitHub Actions CI, and MCP server — zero paid APIs.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green)
![Docker](https://img.shields.io/badge/Docker-containerized-blue)
![Streamlit](https://img.shields.io/badge/Deployed-Streamlit_Cloud-red)
![CI](https://github.com/CharonK652302/Cortex-ai/actions/workflows/ci.yml/badge.svg)

---

## 🚀 Live Demo

🌐 [Try it on Streamlit Cloud](https://cortex-ai-hbgbdge9ldkmkfpnmwforn.streamlit.app/)

---

## 📊 RAG Evaluation Results

| Metric | Score |
|---|---|
| Avg Faithfulness | 0.45 |
| Avg Answer Relevancy | 0.569 |

Evaluated using custom MLflow-tracked evaluation pipeline (`evaluate_rag_mlflow.py`) measuring how grounded and relevant answers are to retrieved context.

- Model: TinyLlama-1.1B / microsoft/phi-2
- Embeddings: sentence-transformers/all-mpnet-base-v2
- Top-k retrieval: 3 chunks
- Experiment tracking: MLflow (2 runs logged)

---

## ✨ Features

- 📂 Upload any PDF and ask natural language questions
- 🤖 Answers grounded strictly in document content
- 🔍 Context-aware retrieval using FAISS vector search
- 📚 Source chunks displayed alongside every answer
- ⚡ Local LLM inference — zero paid APIs
- 🌐 REST API via FastAPI (3 endpoints)
- 🐳 Fully Dockerized for production deployment
- 🔗 MCP server for Claude Desktop / Cursor integration
- 📈 MLflow experiment tracking for RAG evaluation
- ✅ GitHub Actions CI — automated testing on every push
- 🤖 Crew.ai multi-agent research pipeline
- 📑 LlamaIndex VectorStoreIndex PDF Q&A

---

## 🧠 Tech Stack

| Component | Technology |
|---|---|
| LLM | TinyLlama-1.1B / phi-2 (HuggingFace Transformers) |
| Embeddings | sentence-transformers/all-mpnet-base-v2 |
| Vector DB | FAISS |
| RAG Framework | LangChain LCEL + LlamaIndex |
| Multi-Agent | Crew.ai (Researcher + Writer agents) |
| Experiment Tracking | MLflow |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Containerization | Docker |
| CI/CD | GitHub Actions + pytest |
| MCP Server | Model Context Protocol (Claude Desktop) |
| Language | Python 3.11 |

---

## 🏗️ Architecture

```
PDF Upload → Chunking → Embeddings → FAISS Index
                                          ↓
User Query → Similarity Search → Top-k Chunks → LLM → Answer + Sources
                                          ↓
                                    MLflow logs metrics per run
```

---

## 📁 Project Structure

```
cortex-ai/
├── app/
│   └── ui.py                    # Streamlit UI
├── rag/
│   ├── retriever.py             # Original FAISS retriever
│   ├── vector_store.py          # FAISS index builder
│   ├── langchain_retriever.py   # LangChain LCEL RAG pipeline ← NEW
│   └── llamaindex_qa.py         # LlamaIndex VectorStoreIndex ← NEW
├── ingestion/
│   └── ingest_docs.py           # PDF ingestion pipeline
├── tests/
│   └── test_cortex.py           # pytest unit tests ← NEW
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI ← NEW
├── api.py                       # FastAPI REST API
├── crewai_pipeline.py           # Crew.ai multi-agent ← NEW
├── evaluate_rag.py              # Original RAG evaluation
├── evaluate_rag_mlflow.py       # MLflow-tracked evaluation ← NEW
├── dsa_practice.py              # DSA practice (25 problems) ← NEW
├── ragas_results.json           # Evaluation scores
├── Dockerfile
└── requirements.txt
```

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Health check + PDF status |
| POST | `/upload-pdf` | Upload and process PDF |
| POST | `/ask` | Ask question, get answer + sources |

```bash
python api.py
# Docs: http://localhost:8000/docs
```

---

## 🤖 LangChain LCEL RAG Pipeline

```bash
python rag/langchain_retriever.py data/documents/your_doc.pdf
```

Uses LangChain Expression Language (LCEL):
```
retriever | format_docs | RAG_PROMPT | llm | StrOutputParser
```

---

## 📑 LlamaIndex PDF Q&A

```bash
python rag/llamaindex_qa.py data/documents/your_doc.pdf
```

Uses LlamaIndex `VectorStoreIndex` + `VectorIndexRetriever`.

---

## 🤝 Crew.ai Multi-Agent Pipeline

```bash
python crewai_pipeline.py "RAG in LLMs"
```

Two agents working sequentially:
- **Researcher** — finds key facts and insights
- **Writer** — produces a structured technical report

---

## 📈 MLflow Experiment Tracking

```bash
python evaluate_rag_mlflow.py
mlflow ui   # → http://localhost:5000
```

Tracks per run: Faithfulness, Answer Relevancy, Harmonic Mean, top-k, chunk size.

---

## 🔗 MCP Server

Exposes Cortex AI tools to Claude Desktop and Cursor:

```json
{
  "mcpServers": {
    "cortex-ai": {
      "command": "python",
      "args": ["path/to/cortex-mcp-server/server.py"]
    }
  }
}
```

---

## 🐳 Docker

```bash
docker build -t cortex-ai .
docker run -p 8000:8000 cortex-ai
```

---

## ⚙️ Run Locally

```bash
git clone https://github.com/CharonK652302/Cortex-ai.git
cd Cortex-ai
pip install -r requirements.txt
streamlit run app/ui.py   # Streamlit app
python api.py             # FastAPI
```

---

## ✅ GitHub Actions CI

Automated on every push to main:
- pytest unit tests (9 tests)
- ruff linting

```bash
pytest tests/ -v
```

---

## 👨‍💻 Author

**Sai Charan Goud K**
AI/ML Engineer | LangChain · LlamaIndex · Crew.ai · RAG · FastAPI · Docker · MLflow
[GitHub](https://github.com/CharonK652302) · [LinkedIn](https://www.linkedin.com/in/sai-charan-goud-kowlampet-007654284/) · [HuggingFace](https://huggingface.co/CharanGoud652)
