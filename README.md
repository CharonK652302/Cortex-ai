markdown# 🧠 Cortex AI — RAG-Powered PDF Chat App

> Chat with any PDF using AI. Built with RAG, open-source LLMs, 
> FastAPI REST API, and Docker — deployed on Streamlit Cloud.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green)
![Docker](https://img.shields.io/badge/Docker-containerized-blue)
![Streamlit](https://img.shields.io/badge/Deployed-Streamlit_Cloud-red)

---

## 🚀 Live Demo
🌐 [Try it on Streamlit Cloud](cortex-ai-hbgbdge9ldkmkfpnmwforn.streamlit.app)

---

## 📊 RAG Evaluation Results

| Metric | Score |
|---|---|
| Avg Faithfulness | 0.45 |
| Avg Answer Relevancy | 0.569 |

Evaluated using custom evaluation pipeline (`evaluate_rag.py`) 
measuring how grounded and relevant answers are to retrieved context.
- Model: microsoft/phi-2
- Embeddings: sentence-transformers/all-mpnet-base-v2  
- Top-k retrieval: 5 chunks

---

## ✨ Features

- 📂 Upload any PDF and ask natural language questions
- 🤖 Answers grounded strictly in document content
- 🔍 Context-aware retrieval using FAISS vector search
- 📚 Source chunks displayed alongside every answer
- ⚡ Local LLM inference — zero paid APIs
- 🌐 REST API via FastAPI (3 endpoints)
- 🐳 Fully Dockerized for production deployment

---

## 🧠 Tech Stack

| Component | Technology |
|---|---|
| LLM | microsoft/phi-2 (HuggingFace Transformers) |
| Embeddings | sentence-transformers/all-mpnet-base-v2 |
| Vector DB | FAISS |
| Framework | LangChain |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Containerization | Docker |
| Language | Python 3.11 |

---

## 🏗️ Architecture
PDF Upload → Chunking → Embeddings → FAISS Index
↓
User Query → Similarity Search → Top-k Chunks → LLM → Answer + Sources

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Health check + PDF status |
| POST | `/upload-pdf` | Upload and process PDF |
| POST | `/ask` | Ask question, get answer + sources |

Run API locally:
```bash
python api.py
```
API docs: `http://localhost:8000/docs`

---

## 🐳 Docker

```bash
# Build image
docker build -t cortex-ai .

# Run container
docker run -p 8000:8000 cortex-ai
```

---

## ⚙️ Run Locally

```bash
git clone https://github.com/CharonK652302/Cortex-ai.git
cd Cortex-ai
pip install -r requirements.txt

# Run Streamlit app
streamlit run app/ui.py

# Run FastAPI
python api.py
```

---

## 📂 Project Structure
cortex-ai/
├── app/
│   └── ui.py              # Streamlit UI
├── rag/
│   ├── retriever.py       # Vector store loader
│   └── vector_store.py    # FAISS index builder
├── ingestion/
│   └── ingest_docs.py     # PDF ingestion pipeline
├── api.py                 # FastAPI REST API
├── evaluate_rag.py        # RAG evaluation pipeline
├── ragas_results.json     # Evaluation scores
├── Dockerfile             # Docker configuration
├── .dockerignore
└── requirements.txt

---

## 📈 Evaluation

```bash
python evaluate_rag.py
```

Runs faithfulness and answer relevancy scoring across test queries.
Results saved to `ragas_results.json`.

---

## 👨‍💻 Author

**Sai Charan Goud K**  
AI/ML Engineer | RAG · LangGraph · FastAPI · Docker  
[GitHub](https://github.com/CharonK652302) · 
[LinkedIn](https://www.linkedin.com/in/sai-charan-goud-kowlampet-007654284/)
