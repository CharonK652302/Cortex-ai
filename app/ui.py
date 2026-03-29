import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from transformers import pipeline

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import tempfile

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="📄 Chat with PDF", layout="wide")
st.title("📄 Chat with Your PDF (RAG + FLAN)")

# ===============================
# SIDEBAR (NEW)
# ===============================
with st.sidebar:
    st.title("📘 About")
    st.write("Upload a PDF and ask questions using AI.")
    st.write("Built using RAG + HuggingFace + Streamlit")
    st.write("⚡ Lightweight & fast model for deployment")

# ===============================
# FILE UPLOAD
# ===============================
uploaded_file = st.file_uploader("📂 Upload your PDF", type="pdf")

# ===============================
# MODEL
# ===============================
@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="google/flan-t5-small",
        device=-1
    )

generator = load_model()

# ===============================
# EMBEDDINGS
# ===============================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ===============================
# PROCESS PDF
# ===============================
def process_pdf(uploaded_file):

    if uploaded_file.size == 0:
        st.error("❌ Uploaded file is empty.")
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    if not documents:
        st.error("❌ Could not read PDF.")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80
    )

    docs = splitter.split_documents(documents)

    if not docs:
        st.error("❌ No readable content found.")
        return None

    embeddings = get_embeddings()

    try:
        db = FAISS.from_documents(docs, embeddings)
    except Exception as e:
        st.error(f"❌ FAISS error: {str(e)}")
        return None

    return db

# ===============================
# LOAD DB
# ===============================
if uploaded_file:
    db = process_pdf(uploaded_file)

    if db is not None:
        st.success("✅ PDF processed! Ask your questions below.")
    else:
        st.stop()
else:
    st.warning("📂 Upload a PDF to start")
    st.stop()

# ===============================
# CHAT MEMORY
# ===============================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ===============================
# QA FUNCTION
# ===============================
def ask_question(query):

    docs = db.similarity_search(query, k=2)

    context = "\n".join([doc.page_content for doc in docs])[:1500]

    prompt = f"""
Answer the question using ONLY the context.

Context:
{context}

Question:
{query}
"""

    try:
        result = generator(
            prompt,
            max_new_tokens=80,
            do_sample=False
        )

        answer = result[0]["generated_text"].strip()

        # fallback
        if len(answer.strip()) < 5:
            answer = "I couldn't find relevant information in the document."

    except Exception as e:
        answer = f"❌ Model error: {str(e)}"

    # 🔥 ADD PAGE NUMBER (SAFE "highlight")
    sources = [
        f"(Page {doc.metadata.get('page', 'N/A')})\n{doc.page_content[:150]}"
        for doc in docs
    ]

    return answer, sources

# ===============================
# DISPLAY CHAT
# ===============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ===============================
# DOWNLOAD CHAT (NEW)
# ===============================
if st.session_state.messages:
    st.download_button(
        "📥 Download Chat",
        str(st.session_state.messages),
        file_name="chat_history.txt"
    )

# ===============================
# USER INPUT
# ===============================
if query := st.chat_input("Ask something about your PDF..."):

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("🤖 AI is analyzing your document..."):   # ✅ improved UX
            answer, sources = ask_question(query)

            st.write(answer)

            with st.expander("📚 Sources (with page numbers)"):
                for i, src in enumerate(sources):
                    st.code(src)

    st.session_state.messages.append({"role": "assistant", "content": answer})