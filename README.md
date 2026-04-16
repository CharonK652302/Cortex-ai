# 📄 Chat with Your PDF (RAG + LLM)

An AI-powered application that allows users to **interact with PDF documents using natural language queries**.  
Built using **Retrieval-Augmented Generation (RAG)** with open-source LLMs and deployed via Streamlit
---

## 🚀 Features

- 📂 Upload any PDF and ask questions
- 🤖 AI answers based only on document content
- 🔍 Context-aware retrieval using vector search (FAISS)
- 📚 Displays source chunks for transparency
- ⚡ Fast and efficient local inference (TinyLlama)
- 💬 Clean chat-based UI (Streamlit)

---

## 🧠 Tech Stack

- **LLM**: TinyLlama (Hugging Face Transformers)
- **Embeddings**: sentence-transformers (all-mpnet-base-v2)
- **Vector DB**: FAISS
- **Framework**: LangChain
- **Frontend**: Streamlit
- **Language**: Python

---

## 🏗️ Architecture

1. 📄 PDF is loaded and split into chunks  
2. 🔢 Chunks are converted into embeddings  
3. 🗂️ Stored in FAISS vector database  
4. 🔍 User query → similarity search  
5. 🧠 Retrieved context passed to LLM  
6. 💬 Final answer generated  

---

## 📸 Demo

> Upload a PDF → Ask questions → Get accurate answers with sources

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/cortex-ai.git
cd cortex-ai
pip install -r requirements.txt

▶️ Run Locally
streamlit run app/ui.py


🌐 Deployment
Deployed using Streamlit Community Cloud

📂 Project Structure
cortex-ai/
│
├── app/
│   └── ui.py            # Streamlit UI
│
├── rag/
│   ├── retriever.py
│   └── vector_store.py
│
├── ingestion/
│   └── ingest_docs.py
│
├── data/
│   └── documents/       # Sample PDFs
│
├── requirements.txt
├── .gitignore
└── README.md

⚠️ Limitations
Performance depends on CPU (no GPU)
Slower response for large documents
Not optimized for scanned PDFs

🔥 Future Improvements
📌 Highlight answers inside PDF
⚡ Faster inference models
🌐 Multi-document support
🎯 Better UI/UX
☁️ Cloud-based LLM integration

🙌 Acknowledgements
Hugging Face
LangChain
FAISS
Streamlit

👨‍💻 Author
Sai Charan Goud K
AI/ML Engineer | RAG · LangGraph · LLMs
