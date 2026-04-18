import os
import sys
sys.path.append(os.path.abspath("."))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import uvicorn

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

app = FastAPI(
    title="Cortex AI — RAG PDF API",
    description="REST API for RAG-powered PDF question answering",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db = None

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading LLM...")
generator = pipeline(
    "text-generation",
    model="microsoft/phi-2",
    device=-1
)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_retrieved: int

@app.get("/")
def root():
    return {
        "message": "Cortex AI RAG API is running",
        "endpoints": ["/upload-pdf", "/ask", "/health"]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "pdf_loaded": db is not None
    }

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global db

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    db = FAISS.from_documents(chunks, embeddings)

    return {
        "message": "PDF processed successfully",
        "pages": len(documents),
        "chunks": len(chunks),
        "filename": file.filename
    }

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    global db

    if db is None:
        raise HTTPException(
            status_code=400,
            detail="No PDF uploaded yet. Call /upload-pdf first."
        )

    docs = db.similarity_search(request.question, k=5)
    context = "\n".join([doc.page_content for doc in docs])[:1500]

    prompt = f"Answer based on context only.\nContext: {context}\nQuestion: {request.question}\nAnswer:"

    result = generator(prompt, max_new_tokens=150, do_sample=False)[0]["generated_text"]
    result = result.split("Answer:")[-1].strip()

    sources = [
        f"Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:100]}..."
        for doc in docs
    ]

    return AnswerResponse(
        answer=result.strip(),
        sources=sources,
        chunks_retrieved=len(docs)
    )

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)