import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from transformers import pipeline

# PDF + RAG imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import tempfile

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="📄 Chat with PDF", layout="wide")

st.title("📄 Chat with Your PDF (RAG + Phi-2 ⚡)")
st.markdown("### 💬 Chat with your document")

# ===============================
# FILE UPLOAD
# ===============================
uploaded_file = st.file_uploader("📂 Upload your PDF", type="pdf")

# ===============================
# LOAD MODEL (FAST)
# ===============================
@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="microsoft/phi-2",
        device=-1
    )

generator = load_model()

# ===============================
# CACHE EMBEDDINGS
# ===============================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

# ===============================
# PROCESS PDF
# ===============================
def process_pdf(uploaded_file):
    try:
        uploaded_file.seek(0)  # 🔥 FIX empty file issue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        if not documents:
            st.error("❌ Could not read PDF. Try another file.")
            return None

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        docs = splitter.split_documents(documents)

        if not docs:
            st.error("❌ No readable text found (maybe scanned PDF).")
            return None

        embeddings = get_embeddings()

        db = FAISS.from_documents(docs, embeddings)

        return db

    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        return None

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
    st.warning("📂 Please upload a PDF to begin.")
    st.stop()

# ===============================
# CHAT MEMORY
# ===============================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ===============================
# QUESTION FUNCTION (FAST)
# ===============================
def ask_question(query):
    docs = db.similarity_search(query, k=2)  # 🔥 reduced for speed

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer the question using ONLY the context below.

If not found, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

    result = generator(
        prompt,
        max_new_tokens=100,
        do_sample=False   # 🔥 faster
    )[0]["generated_text"]

    answer = result.split("Answer:")[-1].strip()

    sources = [doc.page_content[:200] for doc in docs]

    return answer, sources

# ===============================
# DISPLAY CHAT
# ===============================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ===============================
# USER INPUT
# ===============================
if prompt := st.chat_input("Ask something about your PDF..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = ask_question(prompt)

            st.write(answer)

            with st.expander("📚 View Sources"):
                for i, src in enumerate(sources):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.code(src)

    st.session_state.messages.append({"role": "assistant", "content": answer})